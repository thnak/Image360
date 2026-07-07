# RawIngest — LibRaw Unpack + GPU Demosaic

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first *real* `ITaskExecutor` — `RawIngestExecutor`,
covering `docs/ARCHITECTURE.md` §4.1. Replaces v1's
`ImageLoader::DecodeROI` (full `dcraw_process()` per chunk × image, the
root N×M perf bug) with a one-time-per-image GPU demosaic kernel chain,
keyed by image id and independent of the chunk grid.

**Depends on:** `2026-07-07-task-scheduler-core.md` (needs `ITaskExecutor`),
`2026-07-07-storage-engine.md` (needs `StorageEngine::WritePixelBuffer`).

**Where code lives (important, easy to get backwards):** `WindowsApp.Core`
currently has a `ProjectReference` to `WindowsApp.Compute`, never the
reverse — adding one from Compute back to Core to implement `ITaskExecutor`
there would be a circular project reference. So the concrete
`RawIngestExecutor : public ITaskExecutor` class lives in
**`WindowsApp.Core`**, the same way `DemoStitchExecutor` lives in the
`WindowsApp` UI project (issue #15) rather than in `WindowsApp.Core` — a
concrete executor lives at the *consuming* end of the dependency arrow.
The new CUDA kernels and their C++ façade (the actual demosaic math) live
in `WindowsApp.Compute`, matching `docs/ARCHITECTURE.md` §10's module
table. `RawIngestExecutor` (Core) calls into that façade (Compute); it
does not implement any interface defined in Compute.

**Tech Stack:** existing vendored LibRaw (`WindowsApp.Core/libraw/`,
confirmed fields below), new CUDA kernels in `WindowsApp.Compute`, existing
`StorageEngine`/`ProjectManager` from prior plans.

**LibRaw fields this plan uses (confirmed present in the vendored
`libraw.h`/`libraw_types.h`):** `imgdata.rawdata.raw_image` (unpacked CFA
plane), `imgdata.idata.filters` (CFA pattern; `0` = full-color/Foveon,
non-`9` non-`0` = Bayer, `9` = X-Trans per LibRaw's own convention - verify
against this vendored version rather than assuming), `imgdata.idata.is_foveon`,
`imgdata.idata.xtrans`, `imgdata.color.black`, `libraw_get_cam_mul`,
`libraw_get_rgb_cam`/`cam_xyz_coeff`.

## Global Constraints

- `unpack()` only, never `dcraw_process()`, for the standard-Bayer GPU
  path — that's the entire point of this plan (§4.1's "Why this fixes the
  v1 bottleneck").
- Exotic CFAs (X-Trans, Foveon) fall back to LibRaw's own CPU
  `dcraw_process()` (i.e. `ImageLoader::DecodeFull`, already implemented)
  for demosaic only — a deliberate, documented exception per §4.1, not a
  gap to close in this plan.
- **GPU demosaic output correctness cannot be verified from this
  environment** (no GPU, no Windows) and is called out in
  `docs/ARCHITECTURE.md` §13 as an explicit open risk needing validation
  against LibRaw's reference output before being trusted as the sole
  Bayer path. This plan implements the well-known, standard algorithms
  (nearest-metadata black-level/WB scaling, bilinear Bayer interpolation,
  linear camera→sRGB matrix) faithfully, but flags every kernel's numeric
  correctness as needing a real-GPU manual/visual check — do not claim
  visual correctness was verified.
- One `Task` per image (`unit_kind = 'image'`, `unit_key` = the image's
  DB id as a string) - a crash/cancel mid-ingest loses at most the one
  image being demosaiced, per §4.1's resumability note.

---

### Task 1: `cfa_type` on `input_images` + `InputImageModel`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/Types.h`
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
```cpp
enum class CfaType { BAYER, X_TRANS, FOVEON, UNKNOWN };
```
Add `CfaType cfaType = CfaType::UNKNOWN;` to `InputImageModel` (`ProjectManager.h`).

- [ ] **Step 1: Add the `cfa_type` column**

`ALTER TABLE input_images ADD COLUMN cfa_type TEXT DEFAULT 'BAYER';` run
in both `CreateProject` and `LoadProject` (same "runs on both paths"
pattern as the `tasks`/`chunk_contributors`/`blob_directory` tables from
the `.vfp` schema plan) - `ALTER TABLE ADD COLUMN` fails if the column
already exists, so wrap the call so that failure is silently tolerated
(it means an already-migrated project, not an error) rather than using
`ExecuteNonQuery`'s pass/fail return for this one call.

- [ ] **Step 2: Extend `AddInputImage`/`LoadInputImages`**

`AddInputImage` gains a `CfaType cfaType` parameter (default `CfaType::BAYER`
so existing call sites compile unchanged where the caller doesn't know
yet), stores it as the string `"BAYER"`/`"X_TRANS"`/`"FOVEON"`/`"UNKNOWN"`.
`LoadInputImages` parses that column back into `InputImageModel::cfaType`.

- [ ] **Step 3: Parse check**

Run:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\Types.h,WindowsApp.Core\HeaderFiles\ProjectManager.h -Pattern 'CfaType|cfaType'
```
Expected: matches in both files.

---

### Task 2: `ImageLoader::UnpackRaw` — CFA plane without `dcraw_process()`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ImageLoader.h`
- Modify: `WindowsApp.Core/SourceFiles/ImageLoader.cpp`

**Interfaces:**
```cpp
struct RawPlane
{
    int width = 0;
    int height = 0;
    std::vector<unsigned short> cfaData; // one sample per pixel, raw sensor values
    CfaType cfaType = CfaType::UNKNOWN;
    unsigned blackLevel = 0;
    float camMul[4] = { 1.0f, 1.0f, 1.0f, 1.0f };  // per-channel WB multipliers (libraw_get_cam_mul)
    float rgbCam[3][4] = {};                        // camera RGB -> sRGB matrix (libraw_get_rgb_cam / cam_xyz_coeff)
};

// Stops after LibRaw's unpack() - no dcraw_process(). Must be called on
// an already-Open()'d file. Populates RawPlane::cfaType from
// imgdata.idata.is_foveon / imgdata.idata.filters so callers can route
// Bayer-vs-exotic without duplicating that detection logic themselves.
bool UnpackRaw(RawPlane& output);
```

- [ ] **Step 1: Implement `UnpackRaw`**

In `Impl` (pimpl'd, same as existing methods): call the underlying
LibRaw instance's `unpack()` (not `dcraw_process()`), read
`imgdata.rawdata.raw_image` + `imgdata.sizes.raw_width`/`raw_height` into
`RawPlane::cfaData`/`width`/`height`, `imgdata.color.black` into
`blackLevel`, `libraw_get_cam_mul(lr, i)` for `i` in `0..3` into `camMul`,
`libraw_get_rgb_cam(lr, i, j)` (or `cam_xyz_coeff`, whichever the
vendored version's public API actually exposes cleanly - verify against
`libraw.h` at implementation time) into `rgbCam`.

- [ ] **Step 2: CFA type detection**

```cpp
if (imgdata.idata.is_foveon) output.cfaType = CfaType::FOVEON;
else if (imgdata.idata.filters == 9) output.cfaType = CfaType::X_TRANS;
else if (imgdata.idata.filters != 0) output.cfaType = CfaType::BAYER;
else output.cfaType = CfaType::UNKNOWN;
```
Comment the `filters == 9` constant with a note that it's LibRaw's own
convention, confirmed against this vendored version's headers rather than
assumed from general LibRaw documentation.

- [ ] **Step 3: Error handling**

Same pattern as `DecodeFull`/`DecodeROI`: return `false` and set the
`Impl`'s last-error string on any LibRaw error code, no exceptions.

- [ ] **Step 4: Test coverage**

`WindowsApp.Tests` gains a test that calls `UnpackRaw` on... there is no
committed sample RAW fixture in this repo today. Do not invent one that
doesn't exist - instead, add a test that's skippable/documented as
requiring a fixture: check for a RAW file at a well-known path (e.g.
`WindowsApp.Tests/fixtures/sample.dng`, documented in a short README in
that folder) and `Assert::Inconclusive` (MSTest's designed-for-this
outcome) with a clear message if it's absent, rather than failing the
whole suite or silently skipping. Note in the plan's Self-Review that
sourcing a real small RAW fixture (a few KB, permissively licensed) is
follow-up work, not blocking this plan.

---

### Task 3: GPU demosaic kernel chain (`WindowsApp.Compute`)

**Files:**
- Create: `WindowsApp.Compute/HeaderFiles/demosaic.cuh`
- Create: `WindowsApp.Compute/SourceFiles/demosaic.cu` (or wherever this
  project's existing `.cu` kernel definitions live - mirror
  `median_stack.cuh`'s companion `.cu` file location)
- Modify: `WindowsApp.Compute/HeaderFiles/CudaPipeline.h`
- Modify: `WindowsApp.Compute/WindowsApp.Compute.vcxproj`

**Interfaces:**
```cpp
// demosaic.cuh, namespace WindowsApp::Compute::Kernels - same style as
// median_stack.cuh/tensor_ops.cuh.
__global__ void BlackLevelSubtractKernel(unsigned short* cfaData, int numPixels, unsigned short blackLevel);
__global__ void WhiteBalanceKernel(unsigned short* cfaData, int width, int height, const float camMul[4], int filters);
__global__ void DemosaicBayerKernel(const unsigned short* __restrict__ cfaData, unsigned short* __restrict__ rgbOut, int width, int height, int filters); // bilinear
__global__ void ColorMatrixKernel(unsigned short* __restrict__ rgbData, int numPixels, const float rgbCam[3][4]);
__global__ void ToneCurveKernel(unsigned short* __restrict__ rgbData, int numPixels); // linear passthrough for v1

// CudaPipeline.h - one façade call chaining all five, matching this
// class's existing "one public method per logical operation" style
// (WarpPerspective/MedianStack/MultiBandBlend/ApplyGain).
ComputeResult DemosaicBayer(
    const unsigned short* cfaData, int width, int height,
    unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
    int filters, unsigned short* rgbOut /* width*height*3, pre-allocated */);
```

- [ ] **Step 1: `BlackLevelSubtractKernel`**

One thread per CFA sample: `cfaData[i] = (cfaData[i] > blackLevel) ?
cfaData[i] - blackLevel : 0;` (saturating subtract, matches how
`libraw`'s own CPU path treats black level).

- [ ] **Step 2: `WhiteBalanceKernel`**

One thread per CFA sample: determine which of the (up to 4) Bayer
channels this `(x, y)` belongs to via the same bit-pattern decode LibRaw
uses for `filters` (`(filters >> (((row << 1 & 14) | (col & 1)) << 1)) &
3` - confirmed present in the vendored `libraw.h` as `COLOR(row, col)`;
either call an equivalent inline device function or replicate the
formula, whichever keeps this kernel self-contained without pulling
LibRaw headers into `WindowsApp.Compute`), scale by `camMul[channel]`,
clamp to `unsigned short` range.

- [ ] **Step 3: `DemosaicBayerKernel` (bilinear)**

One thread per output pixel: for each of R/G/B, if this pixel's CFA
channel already *is* that color, use it directly; otherwise bilinear-
average the nearest same-color CFA neighbors (standard 2x2/4-neighbor
bilinear Bayer demosaic - the well-known baseline algorithm;
`docs/ARCHITECTURE.md` §4.1 explicitly scopes v1 to bilinear with
"upgrade path: Malvar-He-Cutler / AHD without changing the surrounding
graph" as later work, not this plan's job).

- [ ] **Step 4: `ColorMatrixKernel`**

One thread per pixel: `outRGB = rgbCam (3x4, homogeneous) * [inR, inG,
inB, 1]`, matching how `cam_xyz_coeff`-derived matrices are conventionally
applied (per-pixel 3x4 linear transform, not per-channel scalar).

- [ ] **Step 5: `ToneCurveKernel`**

v1 scope per §4.1: linear passthrough (no-op transform) - implemented as
a real (trivial) kernel, not skipped, so the graph shape described in §4.1
is genuinely five stages and later plans can swap this one kernel's body
without touching anything else.

- [ ] **Step 6: `CudaPipeline::DemosaicBayer` façade**

Allocates intermediate device buffers (same `cudaMalloc`/`cudaFree`-per-call
style the rest of `CudaPipeline` currently uses - §4.6's persistent-pool
migration is explicitly out of scope for this plan, a separate later
concern), launches the five kernels in sequence on the default stream,
copies the final RGB buffer to `rgbOut`, frees intermediates. Follows the
existing `ComputeResult`/`SetError` error-handling convention used by
`WarpPerspective`/`MedianStack`/etc.

- [ ] **Step 7: Add to `WindowsApp.Compute.vcxproj`**

Register the new `.cuh`/`.cu` files the same way `median_stack.cuh`/
`tensor_ops.cuh` and their `.cu` companions are registered - mirror their
exact `ItemGroup` entries and any CUDA-specific build settings
(`CodeGeneration`, etc.) rather than guessing new ones.

---

### Task 4: `RawIngestExecutor` (`WindowsApp.Core`)

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/RawIngestExecutor.h`
- Create: `WindowsApp.Core/SourceFiles/RawIngestExecutor.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
class RawIngestExecutor : public ITaskExecutor
{
public:
    // Non-owning references - constructed and registered once per
    // PipelineDriver/TaskScheduler setup, matching how DemoStitchExecutor
    // is a lightweight, stateless-ish class constructed per run today.
    RawIngestExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                       std::shared_ptr<Compute::CudaPipeline> cudaPipeline);

    bool Execute(Task& task, CancellationToken token) override;

private:
    ProjectManager& m_projectManager;
    StorageEngine& m_storageEngine;
    std::shared_ptr<Compute::CudaPipeline> m_cudaPipeline;
};
```

- [ ] **Step 1: Resolve the input image**

`task.unitKey` is the image's DB id as a string (`std::stoi`) - look it
up in `m_projectManager.GetInputImages()` (linear scan is fine at this
scale; a project's input image count is not large enough to need an
index). Return `false` if not found (expected-failure: the row may have
been deleted since the task was seeded).

- [ ] **Step 2: Open + unpack**

`ImageLoader loader; loader.Open(image.file_path); ImageLoader::RawPlane plane; loader.UnpackRaw(plane);`
Return `false` on any failure (propagates `ImageLoader::GetLastError()`
into a log, not into `task` itself - `Task` has no error-message field in
this schema; that's an acceptable, documented scope cut, not silently
swallowed - `Execute`'s caller (`TaskScheduler`) already treats a `false`
return as "increment attempt count, retry", which is the existing,
correct behavior for a transient read failure).

- [ ] **Step 3: Route by CFA type**

- `CfaType::BAYER`: call `m_cudaPipeline->DemosaicBayer(...)` (Task 3)
  with `plane`'s fields, producing an RGB48 `PixelBuffer`.
- `CfaType::X_TRANS` / `CfaType::FOVEON`: fall back to the existing CPU
  path - `loader.DecodeFull(pixelBuffer)` (already implemented,
  unchanged) - this is the "deliberate, documented exception" from
  `docs/ARCHITECTURE.md` §4.1, not new code.
- `CfaType::UNKNOWN`: treat as an expected failure (`return false`) -
  don't guess.

- [ ] **Step 4: Commit**

`m_storageEngine.WritePixelBuffer(buffer, "raw_rgb48")` - on success, set
`task.outputBlobId` to the returned blob id and `return true`.
`TaskScheduler`'s caller (from the prior plan) already handles calling
`ProjectManager::CommitTaskOutput(task.taskId, *task.outputBlobId)` after
`Execute` returns `true` - `RawIngestExecutor` does not call
`CommitTaskOutput` itself, matching the ordering contract from
`2026-07-07-storage-engine.md`.

- [ ] **Step 5: Cancellation**

Check `token.stop_requested()` once at entry (same "checked only at the
task's own entry point, not mid-GPU-work" convention as
`StubTaskExecutor`/`DemoStitchExecutor`) - if already requested, return
`false` immediately without doing any work, so a cancelled run doesn't
spend GPU time on a task it's about to discard from its dispatch queue
anyway. Does not check again mid-decode/mid-kernel-chain (§7.2 - in-flight
GPU work is not preemptible and always finishes/commits once started).

- [ ] **Step 6: Add to `WindowsApp.Core.vcxproj`**, header/source
  consistency check (same pattern as every prior plan's final steps).

---

### Task 5: Seed `STAGE0_INGEST` tasks when input images are added

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
```cpp
// Called after AddInputImage (or in a batch after several) - creates one
// PipelineStage::STAGE0_INGEST / unit_kind="image" task per input image
// row that doesn't already have one. Idempotent via CreateTasksIfAbsent's
// existing UNIQUE(stage, unit_kind, unit_key) semantics - safe to call
// again after adding more images without duplicating tasks for images
// already seeded.
bool SeedIngestTasks();
```

- [ ] **Step 1: Implement `SeedIngestTasks`**

For every row in `GetInputImages()`, build a `Task{ stage =
PipelineStage::STAGE0_INGEST, unitKind = "image", unitKey =
std::to_string(image.id) }`, batch them into one
`CreateTasksIfAbsent(...)` call.

- [ ] **Step 2: Test coverage**

Add N input images, call `SeedIngestTasks`, assert `GetTasksForStage(STAGE0_INGEST).size() == N`.
Call it again with no new images, assert the count is still `N` (no
duplicates) - the same `UNIQUE` re-seed test shape from
`2026-07-07-vfp-project-schema.md`, applied to this new call site.

## Self-Review

- Spec coverage: implements `docs/ARCHITECTURE.md` §4.1's full pipeline
  shape (unpack → 5-kernel GPU chain → exotic-CFA CPU fallback →
  `StorageEngine` commit) and its stated resumability granularity
  (one `Task` per image).
- Placeholder scan: no placeholder kernels - `ToneCurveKernel` is a real
  linear-passthrough kernel (matches §4.1's own stated v1 scope), not a
  stub standing in for missing work.
- Known gaps carried forward, not silently dropped: GPU output correctness
  needs manual/reference-image verification (§13, called out throughout
  this plan); no committed test RAW fixture exists yet (Task 2 Step 4);
  CUDA-graph capture and the persistent memory pool (§4.6) are later
  optimizations, not required for this plan's correctness.
