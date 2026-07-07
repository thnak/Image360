# Image360 — Target Architecture (v2 Redesign)

> Status: **design only** — no code in this repo implements this yet. This document
> is the reference for the ground-up rewrite of the processing engine
> (`WindowsApp.Core` / `WindowsApp.Compute`). The existing `WorkflowController` /
> `CudaPipeline` (synchronous, malloc-per-call, CPU-demosaic-per-chunk) is the
> **v1 baseline being replaced**, not a base to extend.

## 1. Scope & Non-Goals

**Target hardware:** exactly one NVIDIA GPU, compute capability 7.5+ (Turing or
newer — required for both tensor cores and the hardware JPEG/NVDEC blocks
nvJPEG relies on). No multi-GPU work distribution, no AMD/Intel GPU support,
no CPU-only fallback rendering path.

**Target OS:** Windows 11 (DirectStorage for Windows requires Windows 10
1909+ in practice, but design against Win11 22H2+ where GPU decompression is
stable). No Linux/macOS.

**Non-goals for this design:**
- Multi-GPU load balancing or NVLink/P2P — single device only.
- A CPU fallback for the demosaic path on standard Bayer sensors (see §4.1 —
  CPU is retained only for exotic CFAs and for RAW *unpacking*, not demosaic).
- Cross-platform storage backend — the container/cache format is
  DirectStorage-shaped and Windows-only.

## 2. Guiding Principles

1. **Decode once, keep it on the GPU.** Every input image is unpacked and
   demosaiced exactly once per project run and stays VRAM/GPU-container
   resident until Stage 3 no longer needs it. This directly replaces the v1
   defect where `ProcessChunk` re-ran a full `dcraw_process()` per (chunk,
   image) pair.
2. **The CPU's job is metadata and orchestration, not pixels.** LibRaw runs
   only far enough to hand off a raw sensor plane; SQLite holds only
   structured metadata; the orchestration thread issues async work and waits
   on GPU/storage completion, it does not touch pixel buffers.
3. **Async by construction, not by bolted-on threads.** Concurrency comes from
   CUDA streams/graphs and DirectStorage queues with events/fences, not from
   spinning up worker threads around an inherently synchronous API (this was
   the flaw in the old triple-buffered-threads proposal — see §12).
4. **One GPU, well used.** No defensive multi-vendor code paths. Optimize
   hard for the single supported device class (persistent memory pools, CUDA
   graphs, tensor cores, nvJPEG hardware blocks).
5. **Every unit of work is small, idempotent, and independently resumable.**
   Nothing runs as one monolithic pass. Work is split into the smallest
   piece that still amortizes GPU launch/transfer overhead (§7), so the
   pipeline can be cancelled or crash at any point and pick back up having
   lost at most one in-flight unit — never more.

## 3. High-Level Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ WindowsApp (UI)                                                         │
│   WinUI 3 / C++/WinRT shell — project view, editor controls, progress   │
└───────────────────────────────┬─────────────────────────────────────────┘
                                 │ thin façade (async, progress callbacks,
                                 │ Cancel())
┌───────────────────────────────▼─────────────────────────────────────────┐
│ WindowsApp.Core — Orchestration                                         │
│  ┌───────────────┐  ┌────────────────┐  ┌─────────────────────────────┐ │
│  │ ProjectManager│  │ PipelineDriver │  │ StorageEngine               │ │
│  │ (.vfp SQLite) │  │ + TaskScheduler│  │ (DirectStorage queues +     │ │
│  │               │  │ (stage state   │  │  .vfpdata container)        │ │
│  │               │  │  machine +     │  │                             │ │
│  │               │  │  task table)   │  │                             │ │
│  └───────────────┘  └───────┬────────┘  └──────────────┬──────────────┘ │
└──────────────────────────────┼──────────────────────────┼────────────────┘
                               │ issues GPU work           │ issues I/O requests
┌──────────────────────────────▼──────────────────────────▼────────────────┐
│ WindowsApp.Compute — Single-GPU Engine (CUDA + D3D12-interop + nvJPEG)   │
│  ┌────────────┐ ┌───────────┐ ┌────────────┐ ┌─────────┐ ┌────────────┐ │
│  │ RawIngest  │ │ Align     │ │ Optimize   │ │ Render  │ │ NvJpegCodec│ │
│  │ (demosaic) │ │ (features,│ │ (gain,     │ │ (warp,  │ │ (preview/  │ │
│  │            │ │  RANSAC)  │ │  tensor BA)│ │  blend) │ │  export)   │ │
│  └────────────┘ └───────────┘ └────────────┘ └─────────┘ └────────────┘ │
│  Shared: CudaMemoryPool · Stream/Graph registry · D3D12↔CUDA interop    │
└───────────────────────────────────────────────────────────────────────────┘
```

## 4. WindowsApp.Compute — GPU Engine

### 4.1 Raw Ingest & GPU Demosaic

**Decision:** demosaic moves to the GPU. LibRaw is retained only for the part
no in-house code should reimplement — parsing 20+ proprietary RAW containers
— and stops at the unpack step.

```
LibRaw::open_file() + unpack()          [CPU, cheap: no dcraw_process()]
   → raw CFA plane (imgdata.rawdata.raw_image) + metadata
     (black level, white balance multipliers, CFA pattern, cam_xyz matrix,
      orientation, dimensions)
   → sensor plane transferred to GPU (DirectStorage read straight into a
     GPU-visible buffer, or pinned-memory H2D copy for in-session-loaded files)
   → CUDA kernel chain (one CUDA graph per distinct image shape):
       1. BlackLevelSubtractKernel
       2. WhiteBalanceKernel        (per-channel gain from LibRaw metadata)
       3. DemosaicBayerKernel       (bilinear for v1; upgrade path: Malvar-
                                      He-Cutler / AHD without changing the
                                      surrounding graph)
       4. ColorMatrixKernel         (camera RGB → linear sRGB via cam_xyz)
       5. ToneCurveKernel           (optional, linear passthrough by default)
   → GPU-resident RGB48 buffer, cached (compressed) via StorageEngine if it
     won't fit resident alongside the rest of the working set (§6).
```

**Exotic CFA fallback:** Fuji X-Trans and Foveon sensors fall back to
LibRaw's own CPU `dcraw_process()` for demosaic only. This is a deliberate,
documented exception, not a gap — reimplementing X-Trans/Foveon demosaic in
CUDA is out of scope for v1. `ImageMetadata` gains a `cfaType` field so
`RawIngest` can route per image.

**Why this fixes the v1 bottleneck:** the old code called `DecodeROI` (full
`dcraw_process()`) once per (chunk × image) pair with no overlap culling.
Moving demosaic to a one-time GPU step, keyed by image id and independent of
the chunk grid, removes the N×M blowup entirely — it's now O(images), not
O(images × chunks).

**Resumability at this stage:** ingest is one `Task` per image (§7) — a
crash or cancel mid-project loses at most the one image currently being
demosaiced, not the whole ingest pass.

### 4.2 Align (Stage 1)

- Feature source: nvJPEG-decoded embedded RAW previews (see §4.5) at low
  res for the first alignment pass — avoids running the full ingest pipeline
  just to get a coarse preview.
- Feature extract/match: GPU kernels (existing candidate: ORB-style or
  FAST+BRIEF — to be selected in a follow-up design pass, not fixed here).
- Homography solve: reuse the existing tensor-core DLT + Cholesky kernels
  (`tensor_ops.cuh`'s `TensorBuildAtA`/`TensorSolveHomography`) inside a
  RANSAC loop launched as a CUDA graph per candidate image pair.
- Task granularity: one `Task` per image for feature extraction, one `Task`
  per candidate image pair for match+RANSAC — pairs that already resolved a
  homography on a previous run are skipped entirely on resume.

### 4.3 Optimize (Stage 2)

Unchanged in spirit from v1's design (gain compensation, color transfer,
Levenberg-Marquardt bundle adjustment via `TensorNormalEquations`), but now
operating purely on GPU-resident buffers produced by Ingest/Align — no LibRaw
or CacheManager calls occur in this stage at all.

Gain compensation and color transfer are per-image `Task`s like the other
stages. Bundle adjustment is the one exception to "small independent task"
(see §7.3) — LM iterations are inherently sequential, so it resumes via
periodic checkpointing instead of task-level parallelism.

### 4.4 Render (Stage 3)

- **Overlap culling is a hard requirement this time**, not a TODO: each
  chunk's contributor list is computed once per project (chunk bounds vs.
  each image's homography-projected footprint) and stored in `.vfp`
  (`chunk_contributors` table, §9), not recomputed by "assume all images
  overlap."
- **CUDA Graph per chunk-shape class.** The per-chunk sequence (warp each
  contributor → gain → median-stack/sigma-clip → multi-band blend) is
  identical in structure across chunks that share the same
  `(width, height, contributorCount)` triple. Capture once per class with
  `cudaStreamBeginCapture`/`cudaGraphInstantiate`, replay with
  `cudaGraphLaunch` for every chunk in that class — this amortizes kernel
  launch overhead across what will typically be hundreds of chunks.
- **Streams:** a small named set — `IngestStream`, `AlignStream`,
  `RenderStream` — with `cudaEvent_t` cross-stream waits (e.g. `RenderStream`
  waits on the `IngestStream` event for a specific image before warping it).
- **Output path:** finished chunk buffers write out through StorageEngine
  (§5), not through the current `CacheManager`'s per-chunk
  `CreateFileW`/`MapViewOfFile` calls.
- **Task granularity:** one `Task` per chunk, exactly as v1's `ChunkStatus`
  resume already did — this part of v1's design was correct and is kept,
  just generalized into the same `Task` table every other stage now uses
  (§7) instead of being Stage 3's own bespoke mechanism.

### 4.5 nvJPEG Integration (v1 codec scope)

Two integration points only for v1 — NVDEC/NVENC and nvCOMP are explicitly
deferred (no 360 video export planned yet; DirectStorage's built-in GDeflate
GPU decompression covers the compression need without nvCOMP for now):

1. **Preview decode:** most RAW files embed a full or half-size JPEG preview
   (EXIF IFD1/embedded preview). Decode it with `nvjpegDecode` on
   `AlignStream` instead of paying for LibRaw's CPU JPEG decode — feeds
   Stage 1 feature extraction without needing a real demosaic pass just to
   align.
2. **Export encode:** the finished stitched panorama's preview/share exports
   (not the archival full-resolution output, which stays a lossless format)
   are encoded via `nvjpegEncodeImage`, GPU-side, avoiding a CPU round-trip
   for what can be gigapixel-scale data.

nvJPEG runs against the same CUDA context/device as everything else — it
accepts an explicit `cudaStream_t`, so it slots into the stream design above
rather than needing its own device management.

### 4.6 Memory Model

- **Persistent pool, not malloc-per-call.** Replace every `cudaMalloc`/
  `cudaFree` pair in the current `CudaPipeline.cpp` with a stream-ordered pool
  allocator (`cudaMallocFromPoolAsync` against a `cudaMemPool_t` sized from
  the VRAM budget in §6). This is the single biggest correctness-adjacent
  change to the compute layer: today every kernel call pays allocation
  latency and forces an implicit sync at the default stream.
- **GPU selection:** enumerate CUDA devices at startup, require compute
  capability ≥ 7.5, pick the one with the most VRAM (ties broken by SM
  count). Resolve its DXGI adapter by LUID so the D3D12 device created for
  DirectStorage (§5) is *guaranteed* to be the same physical GPU — importing
  shared resources across adapters works but silently falls back to a slow
  path, which would defeat the whole point of this design.

## 5. StorageEngine — DirectStorage Subsystem

**Why DirectStorage:** the workload is large sequential/batched reads of
image and chunk data destined for GPU consumption. DirectStorage's queue
model (`IDStorageQueue::EnqueueRequest` × N, single `Submit`, one fence wait)
amortizes per-request overhead across hundreds of chunks, and its GPU-side
GDeflate decompression means compressed cache data never round-trips through
the CPU to be inflated.

**CUDA/D3D12 interop:** our compute stays CUDA — DirectStorage's GPU
decompression path requires an `ID3D12Device`, so `StorageEngine` owns a
minimal D3D12 device purely as a DirectStorage host, never for rendering.
Destination buffers are `ID3D12Resource` objects imported into CUDA via
`cudaImportExternalMemory` (`cudaExternalMemoryHandleTypeD3D12Resource`);
completion is signaled through an `ID3D12Fence` imported as a
`cudaExternalSemaphore`, so CUDA kernels can `cudaWaitExternalSemaphoresAsync`
on a DirectStorage load instead of the orchestrator polling from the CPU.

**Container format:** the large binary blobs (ingested/demosaiced per-image
buffers, rendered chunk tiles) live in one or more sharded side-car files,
`<project-name>.0001.vfpdata`, `<project-name>.0002.vfpdata`, ... (new shard
past ~4 GB) — kept deliberately separate from the small transactional `.vfp`
project file (§9), for the same reason video editors keep a project file
small and a media-cache directory large: frequent transactional metadata
writes (task status updates, checkpoints) must stay fast and crash-safe,
which a multi-gigabyte file does not support well. `CacheManager` (today's
per-chunk `CreateFileW`/MMF class) is retired; `StorageEngine` replaces it
entirely.

**Write-path caveat:** DirectStorage's write support is less mature than its
read path (originally a read-focused, gaming-load-times API). Design the
write side behind a small interface so it can start on conventional
overlapped `WriteFile` and move to DirectStorage writes once verified stable
on the target Windows/driver versions — see §12 open risks.

**Durability ordering (ties directly into resume, §7.2):** a blob is written
to its `.vfpdata` shard and its completion fence signaled *before* the
corresponding task's metadata row in `.vfp` is committed as `COMPLETED`.
If the process dies between those two steps, the blob is simply orphaned
(harmless, garbage-collected later) and the task re-runs from scratch on
resume — never the reverse, which would mark a task done with no data
backing it.

## 6. Memory & VRAM Budget Management

Single-GPU means VRAM is the hard constraint, not compute. `StorageEngine`
and `RawIngest` share a budget: keep the *currently needed* set of
demosaiced input images resident in VRAM (those overlapping the chunk(s)
being rendered right now, per the overlap-culling table in §4.4), evict the
rest back through the GDeflate-compressed container. This is a working-set
policy keyed off the same per-chunk contributor list already computed for
overlap culling — no separate LRU bookkeeping needed, the render schedule
itself tells you what to keep resident next.

This budget also drives the **task-splitting decisions** in §7.4: chunk
size and (rarely) per-image ingest sub-division are chosen so that a single
task's working set fits the budget on the detected card, not hardcoded.

## 7. Work Decomposition, Cancellation & Resume Model

This is the centerpiece of the redesign: **every stage is driven by the same
small `Task` abstraction**, backed by one SQLite table in `.vfp`, so
cancellation and resume are a single mechanism used everywhere instead of
Stage 3's bespoke `ChunkStatus` being the only resumable part of the
pipeline (as in v1).

### 7.1 The `Task` abstraction

```
Task {
  task_id         -- primary key
  stage           -- INGEST | ALIGN_FEATURES | ALIGN_MATCH | OPTIMIZE_GAIN |
                     OPTIMIZE_COLOR | OPTIMIZE_BUNDLE | RENDER
  unit_kind       -- 'image' | 'image_band' | 'pair' | 'chunk' | 'ba_checkpoint'
  unit_key        -- e.g. "img_7", "img_7:band_3", "img_2:img_9", "C_4_2"
  status          -- PENDING | RUNNING | COMPLETED | FAILED | CANCELLED
  attempt_count
  output_blob_id  -- FK into blob_directory; NULL until committed
  checkpoint_json -- small opaque resume state (only used by bundle adjustment)
  updated_at
}
```

A `TaskScheduler` inside `PipelineDriver` runs a stage by querying
`WHERE stage = ? AND status != COMPLETED` and dispatching only what's left —
this is the *entire* resume mechanism. There is no separate "resume code
path"; every run is a resume of whatever the task table says is left to do,
including a completely fresh project (all tasks `PENDING`).

**Idempotency requirement:** a task must be safe to redo from scratch given
only its `unit_key` and the already-committed outputs of tasks it depends
on. No task may have an observable side effect until it commits (§5's
durability ordering) — this is what makes "redo it" always the correct
recovery action, with no special-cased partial-state repair logic needed
anywhere.

### 7.2 Crash and cancel recovery

- **Crash recovery:** on project load, any task still marked `RUNNING` is
  proof of a previous process that died mid-task (only the live process
  ever sets `RUNNING`). These are reset to `PENDING` unconditionally before
  the scheduler starts — `UPDATE tasks SET status = 'PENDING' WHERE status
  = 'RUNNING'`. Combined with idempotency (§7.1) and the durability
  ordering (§5), this is sufficient; no WAL replay or partial-output repair
  is needed.
- **Cancellation:** a single `CancellationToken` (a thin alias over
  `std::stop_token`, matching v1's existing convention) is threaded from
  `PipelineDriver::Start(token)` down through `TaskScheduler::RunStage(stage,
  token)` into each dispatched task. The scheduler checks
  `token.stop_requested()` **only between task dispatches**, never inside a
  running GPU graph or in-flight DirectStorage batch — GPU work is not
  preemptible mid-kernel, so a task that's already launched is always
  allowed to finish and commit normally. Cancelling therefore means "stop
  starting new tasks," not "abort in-flight work," which keeps the
  idempotency/durability story simple: nothing is ever torn down half-done.
  - UI bridging: the WinUI façade's `Cancel()` call simply invokes
    `stop_source.request_stop()`; the façade's async wrapper around
    `PipelineDriver` observes the stage returning `CANCELLED` and completes
    the user-facing operation accordingly.

### 7.3 The one exception: bundle adjustment checkpointing

Per-image and per-chunk tasks are embarrassingly parallel and trivially
resumable by re-running the ones not yet `COMPLETED`. Levenberg-Marquardt
bundle adjustment (Stage 2) is not — each iteration's parameter update
depends on the previous iteration's state, so it cannot be split into
independent tasks. Instead it is a **single task** (`unit_kind =
'ba_checkpoint'`) whose `checkpoint_json` is updated every K iterations with
the current parameter vector, LM damping factor (`lambda`), and iteration
count. Resuming this task means loading the last checkpoint and continuing
the solve rather than restarting it — the only place in the pipeline where
"resume" means "continue a computation" instead of "redo a unit from
scratch."

### 7.4 Splitting work to fit the detected hardware

Task granularity itself is chosen from the VRAM budget (§6) at the point
each stage's task list is generated, not hardcoded:

- **Render chunk size** (today's `ChunkModel::width/height`, hardcoded to
  4096×4096 in v1) is derived from a VRAM probe at project-creation time —
  e.g. 4096×4096 on a 16GB+ card, shrinking to 2048×2048 or 1024×1024 on
  smaller ones, so a chunk's full working set (all contributing warped
  images + intermediate blend buffers) fits the pool from §4.6.
- **Ingest sub-division:** if a single input image's demosaic working set
  doesn't fit alongside the current resident set (unusually large sensors,
  or a very constrained card), `RawIngest` splits it into row-band tasks
  (`unit_kind = 'image_band'`, `unit_key = "img_id:band_idx"`) instead of
  one whole-image task — same table, finer key, no new mechanism.
- Because granularity is derived at project-creation time and stored as
  concrete task rows in `.vfp`, **a project resumed on a different (weaker
  or stronger) GPU than it was created on still works** — the task list
  doesn't change, only how many of its rows happen to already be
  `COMPLETED`. What does *not* change safely mid-project is the chunk grid
  itself (regenerating it with a different size would invalidate already-
  completed render tasks); see §12 for the open question this raises.

## 8. Concurrency & Scheduling Model

One orchestration thread (`PipelineDriver`) drives a stage state machine
(`IDLE → INGEST → ALIGN → OPTIMIZE → RENDER → COMPLETED/CANCELLED/FAILED`,
same shape as today's `PipelineStage` enum). Within a stage, `TaskScheduler`
(§7) issues work async — DirectStorage requests, CUDA graph launches — and
waits on the relevant CUDA event/D3D12 fence rather than blocking a whole OS
thread per pixel operation. A small in-flight window (e.g. N tasks
dispatched ahead of the one being awaited) gives the same I/O/compute
overlap the old triple-buffered-threads proposal was reaching for, but
driven by the task queue instead of hand-rolled threads.

## 9. Project File Format (`.vfp`) & Data Model

A project is two files (plus N data shards):

| File | Contents | Update pattern |
|---|---|---|
| `<name>.vfp` | SQLite database: everything in the table list below | frequent, small, transactional writes (task status, checkpoints) |
| `<name>.NNNN.vfpdata` | Sharded binary blobs: demosaiced image buffers, rendered chunk tiles | large, infrequent, append-mostly writes via StorageEngine |

`.vfp` tables:

| Table | Purpose | Notes |
|---|---|---|
| `project` | total canvas size, chunk size, creation-time GPU/VRAM probe result | chunk size is derived once (§7.4), not recomputed on resume |
| `input_images` | file path, homography, gain, `cfa_type` | `cfa_type` routes GPU vs. LibRaw-CPU demosaic (§4.1) |
| `chunks` | id, x/y offset, width, height | unchanged shape from v1's `ChunkModel` |
| `chunk_contributors` | chunk id → image id | precomputed overlap-culling result (§4.4), replacing v1's never-implemented TODO |
| `tasks` | the unified `Task` table (§7.1) | one row per unit of work across every stage |
| `blob_directory` | blob id → shard file, offset, length, compressed length, format tag | referenced by `tasks.output_blob_id` |

This keeps `ProjectManager`'s existing DB-centric design (input images +
homography + gain, chunk grid) largely intact — it gains the `tasks`,
`chunk_contributors`, and `blob_directory` tables rather than being replaced
wholesale.

## 10. Module / Project Layout Mapping

| Project | Role | Key new/changed types |
|---|---|---|
| `WindowsApp` | UI shell (unchanged role) | wired to `PipelineDriver` via async façade; `Cancel()` → `CancellationToken` |
| `WindowsApp.Core` | Orchestration | `ProjectManager` (+`tasks`/`chunk_contributors`/`blob_directory` tables), new `PipelineDriver` + `TaskScheduler`, new `StorageEngine`; `WorkflowController`/`CacheManager` retired |
| `WindowsApp.Compute` | GPU engine | `RawIngest`, `Align`, `Optimize`, `Render` namespaces; `CudaMemoryPool`; `NvJpegCodec`; D3D12-interop helper shared with `StorageEngine`; existing `tensor_ops.cuh`/`median_stack.cuh` kernels retained and reused |
| `WindowsApp.Tests` | Tests | gains real coverage of `RawIngest` demosaic correctness, `StorageEngine` round-trips, and `TaskScheduler` crash/resume behavior, replacing the current placeholder test |

## 11. Build/Toolchain Additions

- CUDA Toolkit (existing dependency) — widen `CodeGeneration` beyond the
  current hardcoded `compute_89,sm_89` to cover the actual supported range
  (7.5+: `compute_75,sm_75; compute_86,sm_86; compute_89,sm_89; compute_90,sm_90`)
  plus a PTX fallback for forward compatibility, since a user's GPU won't
  always match the dev machine's.
- DirectStorage SDK (`Microsoft.Direct3D.DirectStorage` NuGet, `dstorage.h`).
- nvJPEG (bundled with the CUDA Toolkit — no separate SDK install).
- A minimal D3D12 dependency (`d3d12.h`/`dxgi1_6.h`) purely to host
  DirectStorage — no D3D12 rendering code otherwise.

## 12. Migration Notes From v1

Kept as-is: `ProjectManager`'s SQLite-centric design, the `Types.h` POD
shapes (`Homography`, `ChunkModel`), the existing tensor-core kernels in
`tensor_ops.cuh`, the vendored LibRaw and SQLite amalgamation, and the
`ChunkStatus`-driven resume *concept* for Stage 3 — generalized into the
`Task` table so every stage gets it, not just render.

Retired outright: `CacheManager` (per-chunk MMF files → `StorageEngine` +
`.vfpdata` shards), the current `CudaPipeline`'s per-call
`cudaMalloc`/`cudaMemcpy`/`cudaFree` pattern (→ persistent pool + streams +
graphs), `ImageLoader::DecodeROI`'s full-frame `dcraw_process()`-per-chunk
behavior (→ one-time GPU ingest), and the never-implemented overlap-culling
TODO (→ precomputed `chunk_contributors` table).

The previously filed "Triple-Buffered CUDA Stream Pipeline" proposal
(GitHub issue #1) is superseded by this document: its goal (overlap I/O,
GPU compute, and disk writes for Stage 3) is preserved, but realized through
DirectStorage queues + CUDA streams/events/graphs (§4.4, §5, §8) rather than
four hand-rolled OS threads around an API that has no async primitives to
overlap in the first place.

## 13. Open Risks / Questions

- **DirectStorage write-path maturity** on the exact Windows/driver versions
  this app will ship on — needs a spike before committing Stage 3 output to
  it (§5's write-path caveat).
- **CUDA↔D3D12 interop overhead** for the shared-fence/shared-resource path
  — needs a microbenchmark against plain pinned-memory H2D/D2H before
  assuming it beats the simpler current copy path for smaller projects.
- **GPU demosaic quality parity** with LibRaw's CPU `dcraw_process()` —
  highlight recovery, noise reduction, and camera-specific color science are
  nontrivial to match; v1 GPU demosaic should be validated against LibRaw
  output on a reference image set before it's trusted as the sole path for
  Bayer sensors.
- **VRAM budget sizing** (§6) — needs real numbers from target GPU tiers
  (e.g. 8 GB vs 24 GB cards) to set the resident-working-set policy
  correctly; too aggressive eviction thrashes the GDeflate path, too
  conservative risks OOM on large multi-gigapixel projects.
- **Changing the chunk grid mid-project** (§7.4) — resuming on different
  hardware is safe because task granularity is fixed at creation time, but
  there's no designed path yet for a user who wants to *re-tile* an
  in-progress project (e.g. moving it to a much smaller GPU than it was
  created on) without discarding completed render tasks. Needs a decision:
  disallow it, or design an explicit re-tile migration that maps completed
  old-grid chunks onto the new grid where possible.

## 14. Interface Sketches

Header-level shape only — no bodies, just enough to pin down the contracts
§7 and §8 depend on. Follows the existing codebase's conventions
(`std::wstring` for paths, `std::function` callbacks, pimpl at the
Core/Compute boundary).

### 14.1 Core: `TaskScheduler` and `PipelineDriver`

```cpp
namespace WindowsApp::Core
{
    // Thin alias, not a new concept — matches WorkflowController's existing
    // use of std::stop_token today. Kept as a named alias so call sites read
    // as "cancellation," and so the underlying primitive can be swapped
    // later without touching every signature.
    using CancellationToken = std::stop_token;

    enum class TaskStatus { PENDING, RUNNING, COMPLETED, FAILED, CANCELLED };

    struct Task
    {
        int64_t taskId = 0;
        PipelineStage stage = PipelineStage::IDLE;
        std::string unitKind;               // "image" | "image_band" | "pair" | "chunk" | "ba_checkpoint"
        std::string unitKey;                // "img_7", "img_7:band_3", "img_2:img_9", "C_4_2"
        TaskStatus status = TaskStatus::PENDING;
        int attemptCount = 0;
        std::optional<int64_t> outputBlobId; // set only on commit, see §5 durability ordering
        std::string checkpointJson;           // only meaningful for 'ba_checkpoint'
    };

    // One implementation per stage (RawIngestExecutor, AlignExecutor, ...).
    // TaskScheduler itself is stage-agnostic — it only knows how to look up
    // rows and dispatch them to whichever executor owns that stage.
    class ITaskExecutor
    {
    public:
        virtual ~ITaskExecutor() = default;

        // Contract: must be idempotent (§7.1) — re-invoked with the same
        // Task after a crash must be safe and produce the same committed
        // result. Expected failures (e.g. a corrupt input file) return
        // false and set an error on `task`; exceptions are reserved for
        // programmer errors, not runtime conditions.
        virtual bool Execute(Task& task, CancellationToken token) = 0;
    };

    class TaskScheduler
    {
    public:
        TaskScheduler(ProjectManager& projectManager, StorageEngine& storageEngine);

        void RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor);

        // Runs every task for `stage` not already COMPLETED, honoring
        // `token`. Returns false if the stage didn't fully complete
        // (cancelled, or a task exhausted its retry budget) — never
        // throws for expected failure, only for programmer error.
        bool RunStage(PipelineStage stage, CancellationToken token,
                      std::function<void(const Task&, float stageProgress)> onTaskProgress);

    private:
        // Any row still RUNNING here was left by a process that died
        // mid-task (only a live process ever sets RUNNING) — reset to
        // PENDING unconditionally before dispatch. Called per RunStage,
        // not just once at project load, so resuming mid-stage is safe too.
        void ReclaimStaleRunningTasks(PipelineStage stage);

        // Bounded in-flight window so GPU/DirectStorage work overlaps
        // (§8) without unbounded fan-out; not a thread pool size, a queue
        // depth of outstanding async task dispatches.
        static constexpr int kMaxInFlight = 4;
    };

    class PipelineDriver
    {
    public:
        using ProgressCallback = std::function<void(PipelineStage stage, float overallProgress)>;
        using LogCallback = std::function<void(const std::wstring&)>;

        void Initialize(ProgressCallback onProgress, LogCallback onLog);

        // Drives INGEST -> ALIGN -> OPTIMIZE -> RENDER, skipping stages
        // whose tasks are already fully COMPLETED. Safe to call again
        // after a previous Cancel() or a crash — every call is a resume,
        // including a fresh project where every task is still PENDING.
        bool Run(const std::wstring& projectPath, CancellationToken token);

        PipelineStage GetCurrentStage() const;
        float GetOverallProgress() const;

    private:
        TaskScheduler m_scheduler;
        ProjectManager m_projectManager;
        StorageEngine m_storageEngine;
    };
}
```

Note what's deliberately absent: there is no `Cancel()` method on
`PipelineDriver` itself. Cancellation is caller-owned — whoever starts the
run owns the `std::stop_source` and hands `PipelineDriver::Run` only the
`token` half. This keeps `PipelineDriver` a pure function of "project +
token in, resumed-or-cancelled result out," and pushes stop-source lifetime
to the one place that actually needs it: the UI layer (§14.2).

### 14.2 UI: WinRT Cancellation Bridging

`PipelineDriver::Run` is blocking (it drives the state machine synchronously
on whatever thread calls it) and must not run on the UI thread. The XAML
shell owns the `std::stop_source`, runs the driver on a `std::jthread`, and
marshals progress callbacks back to the UI thread explicitly — WinRT
callbacks never touch XAML objects directly from a background thread.

```cpp
namespace winrt::WindowsApp::implementation
{
    struct MainWindow : MainWindowT<MainWindow>
    {
        winrt::fire_and_forget StartStitchButton_Click(
            IInspectable const&, RoutedEventArgs const&)
        {
            auto lifetime = get_strong();          // keep `this` alive past suspension
            auto dispatcher = DispatcherQueue();    // capture for cross-thread marshaling

            m_stopSource = std::stop_source{};
            m_pipelineThread = std::jthread(
                [this, lifetime, dispatcher, token = m_stopSource.get_token()]
                {
                    m_pipelineDriver.Initialize(
                        [this, dispatcher](PipelineStage stage, float progress)
                        {
                            // Hop back to the UI thread before touching XAML;
                            // the driver's callback fires on m_pipelineThread.
                            dispatcher.TryEnqueue([this, stage, progress]
                            {
                                EditorStatusText().Text(StageToString(stage));
                                ProgressBar().Value(progress * 100.0);
                            });
                        },
                        [](std::wstring const&) { /* log sink */ });

                    m_pipelineDriver.Run(m_projectPath, token);
                });

            co_return;
        }

        void CancelStitchButton_Click(IInspectable const&, RoutedEventArgs const&)
        {
            // Cooperative: in-flight tasks finish and commit normally (§7.2);
            // Run() returns once the current stage's scheduler observes it.
            m_stopSource.request_stop();
        }

    private:
        std::stop_source m_stopSource;
        std::jthread m_pipelineThread;   // destructor auto-requests-stop + joins
        PipelineDriver m_pipelineDriver;
        std::wstring m_projectPath;
    };
}
```

Two points worth calling out:

- **Window close during an active run is handled for free.** `std::jthread`
  requests stop and joins in its own destructor, so `MainWindow` being torn
  down while `m_pipelineThread` is still running triggers the same
  cooperative-cancel path as the Cancel button — there's no separate
  "shutdown while busy" code path to get wrong.
- **Progress marshaling is the one place this differs from a naive port of
  v1's callback style.** `WorkflowController`'s callbacks assumed a
  same-thread caller; here the callback explicitly captures the
  `DispatcherQueue` and re-enters the UI thread via `TryEnqueue` before
  touching any XAML control, because `PipelineDriver::Run` now genuinely
  runs on a background thread for the whole project duration, not just a
  single stage.
