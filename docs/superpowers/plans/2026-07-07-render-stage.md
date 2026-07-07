# Render (Stage 3) — Overlap Culling, Warp/Stack, StorageEngine Output

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 3 (`docs/ARCHITECTURE.md` §4.4) as
`RenderExecutor` (`ITaskExecutor`, `unit_kind = "chunk"`, single kind so
no composite-dispatch is needed here unlike Align/Optimize), with real
overlap culling populating `chunk_contributors` (replacing v1's
never-implemented "assume all images overlap" TODO) and VRAM-budget-driven
chunk sizing at project-creation time.

**Depends on:** `2026-07-07-optimize-stage.md` (needs gain + color-transfer
outputs per image), `2026-07-07-storage-engine.md`.

**Good news carried forward:** the per-chunk kernel sequence
(`WarpPerspective` → `ApplyGain` → `MedianStack`) already exists on
`CudaPipeline` from v1 and needs no new CUDA kernels for its core path —
this plan's real new work is the *orchestration* (overlap culling, chunk
sizing, task seeding, `StorageEngine` output) around kernels that already
work, not new kernel math like `RawIngest`/`Align`/`Optimize` needed.

**Open design note carried forward, not resolved here:**
`docs/ARCHITECTURE.md` §4.4 describes the per-chunk sequence as
"warp → gain → median-stack/sigma-clip → **multi-band blend**," but
`CudaPipeline::MultiBandBlend` is pairwise (`imgA`, `imgB`) while a
chunk's contributor count is N (often > 2) — it doesn't generalize to
this plan's actual inputs without either extending that kernel to N
images or picking a different N-way multi-band strategy. This plan uses
`MedianStack` alone as the final consensus/seam-handling step (a real,
working, well-established technique for this exact scenario - sigma-
clipped consensus across warped overlapping images), and explicitly
defers wiring in multi-band blending as follow-up work (Self-Review) —
not a silently-dropped requirement, a genuinely unresolved mismatch
between the architecture doc's kernel-level description and the actual
kernel signature that exists, flagged for a deliberate decision later.

## Global Constraints

- `chunks.status`/`cache_path` (the v1 columns, still present in the
  schema) are **not** the resume mechanism anymore — the generic `tasks`
  table is, per §4.4's "generalized into the same Task table" note.
  `RenderExecutor` does not need to read or write `chunks.status`/
  `cache_path` at all; treat those columns as vestigial for resume
  purposes (not this plan's job to remove them).
- Overlap culling uses **bounding-box intersection** of each image's
  homography-projected footprint against the chunk rectangle — a
  conservative superset test (may include a genuinely-non-overlapping
  image occasionally, costing one wasted warp; never wrongly *excludes*
  a real contributor, which is the failure direction that would actually
  corrupt output). Exact polygon intersection is a possible follow-up,
  not required for correctness here.
- Named CUDA streams (`IngestStream`/`AlignStream`/`RenderStream`, §8) and
  CUDA-graph capture per chunk-shape-class (§4.4) are cross-stage
  concurrency optimizations, consistent with every prior plan's
  "correctness first" stance — this plan calls `CudaPipeline`'s existing
  synchronous, default-stream façade methods directly.

---

### Task 1: Overlap culling

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/OverlapCulling.h`
- Create: `WindowsApp.Core/SourceFiles/OverlapCulling.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
// Pure function, no I/O - projects `image`'s unit-square footprint
// ([0,0]-[1,1] in normalized homography space, matching how Homography
// is applied elsewhere in this codebase) through its homography into
// output-canvas space, takes the axis-aligned bounding box of the
// projected quad, and tests it against `chunk`'s rectangle.
bool ImageOverlapsChunk(const InputImageModel& image, const ChunkModel& chunk);

// Returns the ids of every input image whose ImageOverlapsChunk is true.
std::vector<int> FindOverlappingImages(const ChunkModel& chunk, const std::vector<InputImageModel>& images);
```

- [ ] **Step 1: `ImageOverlapsChunk`**

Apply `image.homography` to the 4 corners of... clarify the exact
"unit-square footprint" convention against how `WarpPerspective`'s
existing `offsetX`/`offsetY` + inverse-homography backward-mapping
actually places an image in output space (read `CudaPipeline.h`'s
`WarpPerspective` doc comment and, if needed, its `.cu`
implementation before writing this - the two must agree on what
"applying a homography to an image" means in output-canvas coordinates,
or overlap culling will silently disagree with what actually gets
warped). Take the min/max X and Y of the 4 projected corners as the
image's output-space bounding box; overlap = standard AABB-vs-AABB
intersection test against `(chunk.x_offset, chunk.y_offset,
chunk.width, chunk.height)`.

- [ ] **Step 2: `FindOverlappingImages`**

Straightforward loop + filter over `ImageOverlapsChunk` - no spatial
index needed at this project scale (chunk count × image count is not
large enough to need one; a later plan can add one if profiling says
otherwise).

- [ ] **Step 3: Add to `.vcxproj`**, header/source consistency check.

- [ ] **Step 4: Test coverage**

Pure C++, no GPU - fully testable here:
- A chunk and an image whose identity-homography footprint clearly
  overlaps it → `true`.
- A chunk and an image translated far outside it (via a homography with
  a large translation component) → `false`.
- `FindOverlappingImages` with 3 images (2 overlapping, 1 not) returns
  exactly the 2 overlapping ids.

---

### Task 2: VRAM-budget-driven chunk sizing

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
```cpp
// ProjectManager.h - CreateProject gains a chunkSize parameter,
// defaulting to 4096 so every existing call site (including all tests
// from prior plans) keeps compiling unchanged.
bool CreateProject(const std::wstring& dbPath, int totalWidth, int totalHeight, int chunkSize = 4096);

// Free function, no CUDA dependency (takes a plain byte count, not a
// GpuInfo, so ProjectManager.h/.cpp stay Compute-independent) - callers
// that DO have a Compute::GpuInfo (the future project-creation UI,
// PipelineDriver setup) pass gpuInfo.totalMemory here before calling
// CreateProject. Thresholds match the example numbers in
// docs/ARCHITECTURE.md SS7.4 exactly.
int RecommendedChunkSize(uint64_t totalVramBytes);
```

- [ ] **Step 1: `RecommendedChunkSize`**

```cpp
if (totalVramBytes >= 16ull * 1024 * 1024 * 1024) return 4096;
if (totalVramBytes >= 8ull * 1024 * 1024 * 1024)  return 2048;
return 1024;
```

- [ ] **Step 2: Thread `chunkSize` through `CreateProject`'s existing
  chunk-grid loop** (replacing the hardcoded `const int chunkSize =
  4096;` local with the parameter).

- [ ] **Step 3: Test coverage**

`RecommendedChunkSize` at a few boundary values (just under/at/over each
threshold) - pure arithmetic, trivially testable. `CreateProject` with a
non-default `chunkSize` (e.g. `1024`) - assert the generated
`ChunkModel`s all have `width == height == 1024` (except edge chunks
smaller than a full tile, unchanged existing behavior).

---

### Task 3: `RenderExecutor` + task seeding

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/RenderExecutor.h`
- Create: `WindowsApp.Core/SourceFiles/RenderExecutor.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
class RenderExecutor : public ITaskExecutor // unit_kind = "chunk", unit_key = chunk id (e.g. "C_4_2")
{
public:
    RenderExecutor(ProjectManager&, StorageEngine&, std::shared_ptr<Compute::CudaPipeline>);
    bool Execute(Task& task, CancellationToken token) override;
};

// ProjectManager.h - runs overlap culling for every chunk, calls
// SetChunkContributors, and seeds one STAGE3_RENDER/"chunk" task per
// chunk with at least one contributor (chunks with zero contributors
// are skipped entirely - no task, no wasted dispatch, unlike v1's
// ProcessChunk which still ran and just logged "no images overlap").
bool SeedRenderTasks();
```

- [ ] **Step 1: `SeedRenderTasks`**

For every `ChunkModel` in `GetChunks()`: `FindOverlappingImages` (Task 1),
`SetChunkContributors(chunk.id, overlappingIds)` (existing method,
Plan 1), and if `!overlappingIds.empty()`, add a `Task{stage=STAGE3_RENDER,
unitKind="chunk", unitKey=chunk.id}` to a batch passed to
`CreateTasksIfAbsent`.

- [ ] **Step 2: `RenderExecutor::Execute`**

1. `GetChunkContributors(task.unitKey)` - the contributor image ids
   (already resolved by `SeedRenderTasks`, not recomputed here).
2. For each contributor: find its `STAGE2_OPTIMIZE`/`"color"` task's
   `output_blob_id` (same lookup-by-stage-and-unit-kind pattern
   `AlignExecutor`/`OptimizeExecutor` already use), `ReadPixelBuffer` it,
   `WarpPerspective` into chunk-local coordinates (chunk width/height,
   the image's homography, `offsetX`/`offsetY` = the chunk's
   `x_offset`/`y_offset`), then `ApplyGain` using that image's current
   `gain` (from `InputImageModel::gain`, set by Optimize's `"gain"`
   task) on the warped buffer.
3. `MedianStack` across all warped+gain-applied buffers → one consensus
   buffer for the chunk (see this plan's header note on why multi-band
   blending isn't wired in here).
4. `StorageEngine::WritePixelBuffer` the result (`formatTag =
   "rendered_chunk_rgb48"`), set `task.outputBlobId`.
5. Cancellation checked once at entry only, same convention as every
   other executor in this roadmap.

- [ ] **Step 3: Add to `.vcxproj`**, header/source consistency check.

## Self-Review

- Spec coverage: overlap culling is real (not the v1 TODO), chunk sizing
  is VRAM-derived (not hardcoded), task granularity is one-per-chunk with
  empty chunks skipped entirely (an improvement over v1's
  dispatch-then-no-op), output goes through `StorageEngine` (not
  `CacheManager`, which is already retired).
- Placeholder scan: no placeholder kernels — the warp/gain/stack chain is
  the same real, already-working `CudaPipeline` code v1 used, just
  correctly orchestrated with real contributor lists this time.
- Known gaps carried forward, not silently dropped: multi-band blending
  isn't wired in (header note above — the existing kernel's pairwise
  signature doesn't fit N-way contributors as-is); no CUDA-graph capture
  per chunk-shape-class yet (§4.4's stated optimization, deferred
  alongside every other plan's "correctness first" kernels); no named-
  stream cross-stage overlap (§8) yet.
