# Align (Stage 1) — nvJPEG Preview, GPU Features, Tensor-Core RANSAC

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 1 (`docs/ARCHITECTURE.md` §4.2) as two
`ITaskExecutor` kinds — `AlignFeatureExecutor` (`unit_kind = 'image'`) and
`AlignMatchExecutor` (`unit_kind = 'pair'`) — using nvJPEG-decoded
embedded RAW previews for speed (no full ingest pass needed just to
align), GPU feature extraction/matching, and the existing tensor-core
homography solve.

**Depends on:** `2026-07-07-raw-ingest.md` (needs `CfaType`/`RawIngestExecutor`
established patterns and, for real projects, `STAGE0_INGEST` completed
first so `input_images` rows exist — though Align's *feature* extraction
only needs the embedded JPEG preview, not the full demosaiced image, so
it can technically run in parallel with ingest; this plan still lists
ingest as a dependency because both need the same `RawIngestExecutor`-
adjacent LibRaw-open plumbing established there).

**Concrete algorithm choice this plan makes (deferred by
`docs/ARCHITECTURE.md` §4.2 itself — "to be selected in a follow-up
design pass, not fixed here"):** FAST corner detection + BRIEF binary
descriptors + brute-force Hamming-distance matching with Lowe's ratio
test. Chosen over full ORB (which adds scale pyramids + orientation
compensation) because it's the simplest well-documented GPU-parallel
option that still gives usable correspondences for a first working
alignment path — `docs/ARCHITECTURE.md` §4.2 explicitly names "FAST+BRIEF"
as one of its two candidates. Upgrading to ORB or a learned feature
matcher later does not change `AlignFeatureExecutor`/`AlignMatchExecutor`'s
task granularity or `ITaskExecutor` contract, the same "swap the
internals, not the interface" pattern used throughout this roadmap.

**Where code lives:** same rule as `2026-07-07-raw-ingest.md` — CUDA
kernels/façades in `WindowsApp.Compute`, the concrete `ITaskExecutor`
classes in `WindowsApp.Core` (Compute cannot depend on Core).

**Tech Stack:** nvJPEG (bundled with CUDA Toolkit, per
`docs/ARCHITECTURE.md` §11 — needs `nvjpeg.h`/`nvjpeg.lib` wired into
`WindowsApp.Compute.vcxproj`, not yet present), existing
`tensor_ops.cuh` (`TensorBuildAtA`/`TensorSolveHomography`, already
exposed via `CudaPipeline::TensorEstimateHomography`), LibRaw's
`unpack_thumb()`/`unpack_thumb_ex()` (confirmed present in the vendored
`libraw.h`, not currently called anywhere in `ImageLoader`).

## Global Constraints

- RANSAC in this plan is **host-orchestrated** — a C++ loop on
  `WindowsApp.Core`'s side that calls `CudaPipeline::TensorEstimateHomography`
  once per iteration on a random 4-point sample, scores inliers on the
  CPU, and keeps the best result over a **fixed** iteration count (no
  early termination). `docs/ARCHITECTURE.md` §4.2 describes RANSAC
  "launched as a CUDA graph per candidate image pair," which implies a
  fully GPU-resident sample+solve+score loop — that's a real follow-up
  optimization, not this plan's scope; ship a correct host-orchestrated
  version first, matching the "correctness before CUDA-graph capture"
  approach `2026-07-07-raw-ingest.md` already took for the demosaic chain.
- Feature/match GPU correctness (like RawIngest's demosaic output) cannot
  be verified from this environment — flag it, don't claim it.
- Task granularity per `docs/ARCHITECTURE.md` §4.2: one `Task` per image
  for feature extraction, one `Task` per **candidate** image pair for
  match+RANSAC. This plan does not solve "which pairs are candidates" via
  anything smarter than all-pairs for the seed step (Task 6) — a spatial/
  sequence-adjacency heuristic to avoid O(n²) pairs on large projects is
  explicitly out of scope, noted in Self-Review as a follow-up.

---

### Task 1: nvJPEG build wiring + `NvJpegCodec` decode façade

**Files:**
- Create: `WindowsApp.Compute/HeaderFiles/NvJpegCodec.h`
- Create: `WindowsApp.Compute/SourceFiles/NvJpegCodec.cpp`
- Modify: `WindowsApp.Compute/WindowsApp.Compute.vcxproj`

**Interfaces:**
```cpp
namespace WindowsApp::Compute
{
    class COMPUTE_API NvJpegCodec
    {
    public:
        NvJpegCodec();
        ~NvJpegCodec();
        ComputeResult Initialize(); // creates an nvjpegHandle_t against the current CUDA context
        void Shutdown();

        // Decodes a JPEG byte buffer to interleaved RGB8, allocating
        // `outWidth`/`outHeight`/`outRgb` (caller-owned, matches this
        // class's malloc-per-call style, same scope-cut rationale as
        // CudaPipeline's existing kernels - a pool is later work).
        ComputeResult Decode(const unsigned char* jpegData, size_t jpegSize,
                              unsigned char** outRgb, int* outWidth, int* outHeight);

        void FreeDecoded(unsigned char* rgb);

        const char* GetLastError() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
```

- [ ] **Step 1: Link nvJPEG**

Add `nvjpeg.lib` to `AdditionalDependencies` and confirm
`$(CudaToolkitIncludeDir)`/`$(CudaToolkitLibDir)` (already present for
`cuda_runtime.h`, per the existing `.vcxproj`'s CUDA build customization
imports) also resolve `nvjpeg.h` — it ships in the same CUDA Toolkit
install, no separate SDK download per §11.

- [ ] **Step 2: Implement `Initialize`/`Decode`/`Shutdown`**

Standard nvJPEG single-image decode sequence:
`nvjpegCreateSimple` → `nvjpegJpegStateCreate` → `nvjpegGetImageInfo`
(to size the output buffer) → `cudaMalloc` the output → `nvjpegDecode`
(output format `NVJPEG_OUTPUT_RGBI`) → `cudaMemcpy` device→host into a
caller-visible buffer (host-visible output matches this plan's other
façades returning CPU-side buffers; keeping the result GPU-resident for
the whole Align pipeline is a later optimization once the surrounding
feature kernels also stay GPU-resident end to end) → destroy state on
`Shutdown`.

- [ ] **Step 3: Add to `.vcxproj`**, header/source consistency check.

---

### Task 2: Embedded preview extraction (`ImageLoader`)

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ImageLoader.h`
- Modify: `WindowsApp.Core/SourceFiles/ImageLoader.cpp`

**Interfaces:**
```cpp
// Extracts the embedded preview JPEG bytes (EXIF IFD1 / LibRaw thumbnail)
// without touching the full-res CFA plane. Populated buffer's format is
// always JPEG for this call to succeed (non-JPEG embedded thumbnails,
// e.g. raw bitmap thumbnails on some older cameras, return false - an
// expected, documented gap, not silently mishandled).
bool GetEmbeddedPreviewJpeg(std::vector<unsigned char>& jpegBytes);
```

- [ ] **Step 1: Implement via `unpack_thumb()`**

Call the underlying LibRaw instance's `unpack_thumb()`, check
`imgdata.thumbnail.tformat == LIBRAW_THUMBNAIL_JPEG` (verify this exact
enumerator name against the vendored `libraw_types.h` at implementation
time), copy `imgdata.thumbnail.thumb`/`imgdata.thumbnail.tlength` into
`jpegBytes`. Return `false` (not throw) for any other format or LibRaw
error, same convention as every other `ImageLoader` method.

- [ ] **Step 2: Test coverage**

Same fixture-gated pattern as `2026-07-07-raw-ingest.md` Task 2 Step 4 —
`Assert::Inconclusive` with a clear message if no fixture RAW file is
present, rather than a hard failure or a silent no-op test.

---

### Task 3: FAST + BRIEF feature kernels (`WindowsApp.Compute`)

**Files:**
- Create: `WindowsApp.Compute/HeaderFiles/features.cuh`
- Create: `WindowsApp.Compute/SourceFiles/features.cu`
- Modify: `WindowsApp.Compute/HeaderFiles/CudaPipeline.h`
- Modify: `WindowsApp.Compute/WindowsApp.Compute.vcxproj`

**Interfaces:**
```cpp
// features.cuh, namespace WindowsApp::Compute::Kernels
struct FeaturePoint { float x, y; };
using BriefDescriptor = uint64_t[4]; // 256-bit binary descriptor

__global__ void FastDetectKernel(
    const unsigned char* __restrict__ grayImage, int width, int height,
    FeaturePoint* __restrict__ outPoints, int* __restrict__ outCount, int maxPoints,
    unsigned char threshold);

__global__ void BriefDescribeKernel(
    const unsigned char* __restrict__ grayImage, int width, int height,
    const FeaturePoint* __restrict__ points, int numPoints,
    BriefDescriptor* __restrict__ outDescriptors);

// CudaPipeline.h façade
ComputeResult DetectAndDescribeFeatures(
    const unsigned char* rgbImage, int width, int height, // from NvJpegCodec::Decode
    FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints);
```

- [ ] **Step 1: Grayscale conversion**

A small existing-style kernel (`RgbToGrayKernel`, standard luma weights
`0.299R + 0.587G + 0.114B`) - FAST/BRIEF both operate on single-channel
intensity.

- [ ] **Step 2: `FastDetectKernel`**

Standard FAST-9 (9-of-16 contiguous pixels on the Bresenham circle
brighter or darker than center ± threshold) - one thread per candidate
pixel, atomic-increment into `outCount` (capped at `maxPoints`) on a
positive detection. No non-maximum suppression in this first pass (a
documented simplification, not silently dropped — see Self-Review).

- [ ] **Step 3: `BriefDescribeKernel`**

Standard BRIEF: a fixed, precomputed (host-side constant, uploaded once
via `cudaMemcpyToSymbol` or passed as a kernel argument) set of 256 pixel-
pair offsets within a small patch (e.g. 31x31) around each feature point;
one thread per feature point, each bit of the 256-bit descriptor is
`intensity(pair.a) < intensity(pair.b)`.

- [ ] **Step 4: `CudaPipeline::DetectAndDescribeFeatures` façade**

Chains grayscale → `FastDetectKernel` → `BriefDescribeKernel`, same
malloc-per-call/`ComputeResult` conventions as the rest of this class.

- [ ] **Step 5: Add to `.vcxproj`**, header/source consistency check.

---

### Task 4: Descriptor matching kernel

**Files:**
- Modify: `WindowsApp.Compute/HeaderFiles/features.cuh`
- Modify: `WindowsApp.Compute/SourceFiles/features.cu`
- Modify: `WindowsApp.Compute/HeaderFiles/CudaPipeline.h`

**Interfaces:**
```cpp
struct MatchResult { int indexA; int indexB; int hammingDistance; };

__global__ void BruteForceMatchKernel(
    const BriefDescriptor* __restrict__ descA, int countA,
    const BriefDescriptor* __restrict__ descB, int countB,
    MatchResult* __restrict__ outMatches, int* __restrict__ outMatchCount,
    float ratioThreshold); // Lowe's ratio test, e.g. 0.75

// CudaPipeline.h façade
ComputeResult MatchFeatures(
    const BriefDescriptor* descA, int countA,
    const BriefDescriptor* descB, int countB,
    MatchResult* outMatches, int* outMatchCount, int maxMatches,
    float ratioThreshold = 0.75f);
```

- [ ] **Step 1: `BruteForceMatchKernel`**

One thread per descriptor in A: compute Hamming distance (`__popcll` on
each of the 4 `uint64_t` words, summed) to every descriptor in B, track
best and second-best; accept the match only if
`best / max(second_best, 1) < ratioThreshold` (Lowe's ratio test),
atomic-append to `outMatches`.

- [ ] **Step 2: `CudaPipeline::MatchFeatures` façade + `.vcxproj`/consistency
  check** (same pattern as every prior task).

---

### Task 5: Host-orchestrated RANSAC + homography persistence

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/RansacHomography.h`
- Create: `WindowsApp.Core/SourceFiles/RansacHomography.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
// RansacHomography.h
struct RansacResult { Homography homography; int inlierCount; bool success; };

// Fixed-iteration host-orchestrated RANSAC (see Global Constraints) -
// samples 4 random correspondences per iteration, calls
// CudaPipeline::TensorEstimateHomography, scores inliers (reprojection
// error < inlierThresholdPx) on the CPU, keeps the best-scoring result.
RansacResult RunRansacHomography(
    Compute::CudaPipeline& cudaPipeline,
    const std::vector<std::pair<FeaturePoint, FeaturePoint>>& correspondences,
    int iterations = 500, float inlierThresholdPx = 3.0f);

// ProjectManager.h - new method, this plan's reason for existing:
// AddInputImage only sets the *initial* (identity) homography; Align is
// the first stage that ever needs to overwrite it with a computed one.
bool UpdateHomography(int imageId, const Homography& h);
```

- [ ] **Step 1: `RunRansacHomography`**

Fixed loop of `iterations` (default 500, matching typical classical-CV
RANSAC iteration counts for a 4-point model at moderate outlier
fractions - not empirically tuned against this codebase's own data,
flagged for follow-up tuning once real projects exist to test against):
sample 4 correspondences (`std::mt19937`, not `rand()` - matches modern
C++ conventions this codebase already uses elsewhere, e.g.
`std::stop_token`), call `TensorEstimateHomography`, reproject every
correspondence's point A through the candidate homography, count inliers
within `inlierThresholdPx`, keep the best.

- [ ] **Step 2: `ProjectManager::UpdateHomography`**

`sqlite3_bind_*` prepared statement (matches Plan 1's established style
for the CRUD added since — not the older `snprintf` style
`AddInputImage`/`UpdateChunkStatus` still use), updates the 9
`h00..h22` columns for the given `imageId`, updates the in-memory
`m_inputImages` cache entry to match (same "update the local cache after
a successful write" pattern `UpdateImageGain` already follows).

- [ ] **Step 3: Test coverage**

`RunRansacHomography` against a **synthetic** correspondence set (known
ground-truth homography applied to a handful of points, plus a few
deliberately-wrong outlier pairs) - this *is* testable without a GPU only
if `TensorEstimateHomography` itself is mockable, which it currently
isn't (it's a concrete `CudaPipeline` method, not behind an interface).
Do not introduce a new abstraction layer just to make this unit-testable
in this plan - instead, document plainly in Self-Review that
`RunRansacHomography`'s orchestration logic is exercised end-to-end only
via `AlignMatchExecutor` on real hardware (Task 6), and add a narrower
CPU-only test for just the *inlier-scoring* helper (reprojection error
computation given a known homography and points) factored out as its own
free function specifically so it doesn't require CUDA to test.

---

### Task 6: `AlignExecutor` (composite) + task seeding

**Why one class, not two:** `TaskScheduler::RegisterExecutor` (from
`2026-07-07-task-scheduler-core.md`, already shipped and tested) maps
**one** `ITaskExecutor` per `PipelineStage` — registering a second
executor for `STAGE1_ALIGN` would silently overwrite the first in
`TaskScheduler`'s internal map, not run both. Rather than change that
already-tested public contract, `AlignExecutor` is a single class whose
`Execute` dispatches on `task.unitKind` ("image" → feature extraction,
"pair" → match+RANSAC). This is the pattern every future multi-unit-kind
stage in this roadmap should follow (Optimize, next plan, needs the same
thing for "gain"/"color"/"ba_checkpoint").

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/AlignExecutor.h`
- Create: `WindowsApp.Core/SourceFiles/AlignExecutor.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
class AlignExecutor : public ITaskExecutor // registered once for STAGE1_ALIGN
{
public:
    AlignExecutor(ProjectManager&, StorageEngine&, std::shared_ptr<Compute::CudaPipeline>,
                  std::shared_ptr<Compute::NvJpegCodec>);

    bool Execute(Task& task, CancellationToken token) override; // dispatches by task.unitKind

private:
    bool ExecuteFeatureExtraction(Task& task, CancellationToken token); // unit_kind == "image"
    bool ExecuteMatch(Task& task, CancellationToken token);             // unit_kind == "pair"
    // ...
};

// ProjectManager.h
bool SeedAlignTasks(); // one STAGE1_ALIGN/"image" task per input image,
                        // one STAGE1_ALIGN/"pair" task per all-pairs
                        // combination - see Global Constraints on the
                        // all-pairs scope cut.
```

- [ ] **Step 1: `Execute` dispatch**

```cpp
bool AlignExecutor::Execute(Task& task, CancellationToken token)
{
    if (task.unitKind == "image") return ExecuteFeatureExtraction(task, token);
    if (task.unitKind == "pair")  return ExecuteMatch(task, token);
    return false; // unknown unit_kind is a programmer/data error, not
                   // retryable - still returns false (not throw) per the
                   // ITaskExecutor contract's "expected failures only"
                   // rule, since a malformed row is plausible operator
                   // error (e.g. hand-edited DB), not a crash-worthy bug.
}
```

- [ ] **Step 2: `ExecuteFeatureExtraction`**

`GetEmbeddedPreviewJpeg` → `NvJpegCodec::Decode` → `CudaPipeline::DetectAndDescribeFeatures`
→ persist points+descriptors via `StorageEngine::WriteBlob` (raw bytes,
`formatTag = "features_brief256"`) → `task.outputBlobId` = that blob id.

- [ ] **Step 3: `ExecuteMatch`**

Parse `task.unitKey` (`"img_A:img_B"`) into two image ids, look up each
image's completed feature task's `output_blob_id` via
`GetTasksForStage(STAGE1_ALIGN)` filtered to `unit_kind == "image"` (an
in-process lookup, not a new `ProjectManager` method - this executor
already has a `ProjectManager&`), `StorageEngine::ReadBlob` both, run
`MatchFeatures` then `RunRansacHomography`, and on success call
`m_projectManager.UpdateHomography(imageB, result.homography)` (image A
stays the reference frame in this v1 pass - a documented simplification;
proper bundle-adjustment-driven global alignment is Stage 2's job per
§4.3, not Align's).

- [ ] **Step 4: `SeedAlignTasks`**

One `unit_kind = "image"` task per input image; one `unit_kind = "pair"`
task per `(i, j)` combination with `i < j` over all input images
(all-pairs — see Global Constraints for the explicit O(n²) scope cut).

- [ ] **Step 5: Add to `.vcxproj`**, header/source consistency check.

## Self-Review

- Spec coverage: implements `docs/ARCHITECTURE.md` §4.2's stated pipeline
  (nvJPEG preview → GPU feature extract/match → tensor-core RANSAC) and
  its task granularity (per-image features, per-pair match+RANSAC).
- Placeholder scan: no placeholder kernels; FAST/BRIEF/brute-force-match
  are complete, if simplified (no non-max suppression, no scale
  invariance), implementations - explicitly named simplifications, not
  hidden gaps.
- Known gaps carried forward: all-pairs task seeding is O(n²) and will
  need a spatial/sequence heuristic before large projects are practical;
  no non-maximum suppression on FAST detections (may over-detect
  clustered corners); RANSAC is fixed-iteration/host-orchestrated, not
  yet the CUDA-graph-captured version §4.2 describes; GPU feature/match
  numeric correctness needs real-hardware verification, same caveat as
  `2026-07-07-raw-ingest.md`.
