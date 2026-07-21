# MFNR: Block-Match Alignment + Robust Merge (Phase 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 1 of `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §8 — the first real
burst-mode algorithm. Implements `BlockMatchAlign` (shared by all four burst
modes) and `RobustMergeAccumulate` (MFNR's merge specifically — HDR+'s
FFT/Wiener merge and Super-Res-Zoom/Night-Sight's kernel-regression merge
are each their own later phase, per §2.3's correction that these are
different algorithms, not parameter variants of one merge kernel). Wires a
complete, working MFNR pipeline end to end: add burst frames → align → merge
→ (identity finish) → readable output blob, driven entirely through the
`Task`/`TaskScheduler`/`PipelineDriver` machinery from Phase 0
(`2026-07-21-burst-pipeline-foundation.md`).

**Depends on:** Phase 0 (`ProjectType::BURST`, `BurstMode`,
`BURST_ALIGN`/`BURST_MERGE`/`BURST_FINISH` stages, `PipelineDriver` stage
sequencing) — already merged.

**Architecture:**
- New `IComputeBackend` ops, per `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §3/§4:
  CPU-first, explicit tracked gap elsewhere. `CpuComputeBackend` gets real
  scalar implementations; `CudaPipeline`/`VulkanPipeline` get a new
  `ComputeResult::NOT_SUPPORTED` stub — a visible, typed "not yet
  implemented" signal instead of a silent wrong answer, a link failure, or
  an invisible gap (the exact failure mode §4's HAL research flagged in
  darktable's history). No AVX2/AVX512 tiers yet either — like
  `HomographyMath`/`LinearSolve`/`OverlapCulling`, this starts as
  plain scalar C++ and only earns SIMD tiers if profiling later shows it's
  hot, matching this codebase's own precedent (not every kernel is
  3-tiered, only the ones that turned out to need it).
- **`BlockMatchAlign`**: non-overlapping tile grid, brute-force integer-pixel
  SAD search within a fixed radius — the same choice HDR+'s own finest
  pyramid level makes ("outperformed by a well-implemented brute-force
  procedure" vs. phase correlation, per `docs/COMPUTATIONAL_PHOTOGRAPHY.md`
  §2.1). **Explicit scope cuts vs. the full papers**: single-level (no
  Gaussian pyramid — HDR+/Wronski use one for larger/coarser motion this
  MVP doesn't attempt to handle), integer-pixel only (no closed-form
  sub-pixel quadratic fit). Both are real limitations for handheld bursts
  with motion beyond the search radius or needing sub-pixel accuracy — not
  hidden, tracked in §9 below as follow-up work, not silently pretended
  complete.
- **`RobustMergeAccumulate`**: per-pixel, reference-frame-relative Gaussian
  robustness weighting — `weight_k = exp(-(sample_k - sample_ref)² /
  (2·sigma²))`, reference frame fixed at weight 1.0, output is the
  weighted average. This mirrors the *spirit* of HDR+'s real algorithm
  (reject each alternate frame's contribution relative to the reference,
  per §2.1) in the spatial domain instead of HDR+'s actual per-frequency-
  bin Wiener shrinkage — a deliberate, documented simplification, not an
  attempt to reproduce HDR+'s FFT merge (that's Phase 2's job, a genuinely
  different kernel). `sigma` is a fixed noise-model parameter (not
  per-ISO-calibrated) — another explicit scope cut, tracked in §9.
- **Frame storage reuses `input_images`/`AddInputImage`** — a burst
  project's "frames" are stored exactly like a panorama project's "input
  images" (file path, gain, `cfa_type`); no new table. Frame 0 (the first
  added) is always the alignment/merge reference.
- **Decode logic is shared with `RawIngestExecutor`**, extracted into a
  free function rather than duplicated — `RawIngestExecutor::Execute`'s
  entire dispatch-by-`CfaType` body becomes `DecodeInputImage(...)`, called
  by both `RawIngestExecutor` and the new `BurstAlignExecutor`. Pure
  refactor, no behavior change to the panorama path (Task 1 below is a
  regression check on this specifically).
- **`BURST_FINISH` is an identity passthrough for MFNR in this phase** —
  copies `BURST_MERGE`'s output blob forward as the task's own output. Real
  finish-stage processing (sharpen, chroma denoise cleanup) is out of scope
  here; MFNR has no tone-mapping need (unlike HDR+/Night Sight), so
  "passthrough" is a real, if minimal, correct implementation, not a stub.
- **All burst tasks seed upfront**, unlike panorama's `STAGE3_RENDER`
  (which needs `chunk_contributors` computed from Optimize's *results*).
  `BURST_MERGE`/`BURST_FINISH` only need `BURST_ALIGN` to have *run*, which
  `TaskScheduler`'s stage ordering already guarantees — no result-dependent
  seeding, so `PipelineDriver::Run`'s loop needs no new special case
  (unlike the `STAGE3_RENDER` branch it already has for panorama).

**Tech Stack:** C++20, existing `WindowsApp.Core` structure;
`tests/pipeline_e2e_burst` (new, CMake/ctest) for the end-to-end quality
gate, generating synthetic JPEG fixtures in-process via `JpegCodec::Encode`
(no committed binary fixtures, no RAW/DNG needed — this phase's frames are
`CfaType::STANDARD_RGB`, sidestepping the DNG-fixture complexity
`tests/pipeline_e2e`'s own history flagged as fiddly).

## Global Constraints

- No CUDA/Vulkan kernel implementations in this plan — `NOT_SUPPORTED`
  stubs only, tracked as a real gap (§9), not silently absent.
- No pyramid/sub-pixel alignment, no calibrated per-ISO noise model — both
  explicit, tracked scope cuts (§9), not silently pretended complete.
- Don't touch the panorama path's behavior — `DecodeInputImage`'s extraction
  must be a pure refactor (Task 1's own regression check enforces this).

---

### Task 1: Extract `DecodeInputImage` from `RawIngestExecutor`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/RawIngestExecutor.h`
- Modify: `WindowsApp.Core/SourceFiles/RawIngestExecutor.cpp`

**Interfaces:**
- Produces: a free function (not a class method — needs no executor state)
  `bool DecodeInputImage(const InputImageModel& image,
  std::shared_ptr<Compute::IComputeBackend> computeBackend,
  std::shared_ptr<Compute::IImageCodec> jpegCodec, PixelBuffer& outBuffer)`
  in `RawIngestExecutor.h`/`.cpp`, containing exactly the body currently
  inline in `RawIngestExecutor::Execute` (the `CfaType` dispatch: STANDARD_RGB
  → JPEG/stb_image, BAYER → GPU demosaic via `computeBackend`, X_TRANS/FOVEON
  → LibRaw CPU `dcraw_process()`).

- [ ] **Step 1: Extract the function**, keeping `RawIngestExecutor::Execute`
  as a thin wrapper: look up the `InputImageModel` by `task.unitKey`, call
  `DecodeInputImage`, `WritePixelBuffer` + set `task.outputBlobId` on
  success — identical behavior, just delegated.
- [ ] **Step 2: Regression check** — the existing `tests/pipeline_e2e`
  scenarios (which exercise `RawIngestExecutor` for real, including the
  X-Trans/Foveon and STANDARD_RGB paths) must still pass unmodified. This
  is the actual proof the extraction changed nothing.

---

### Task 2: `IComputeBackend::BlockMatchAlign` + `RobustMergeAccumulate`

**Files:**
- Modify: `WindowsApp.Compute/HeaderFiles/ComputeTypes.h`
- Modify: `WindowsApp.Compute/HeaderFiles/IComputeBackend.h`
- Modify: `WindowsApp.Compute/HeaderFiles/CudaPipeline.h`,
  `WindowsApp.Compute/SourceFiles/CudaPipeline.cpp`
- Modify: `WindowsApp.Vulkan/HeaderFiles/VulkanPipeline.h`,
  `WindowsApp.Vulkan/SourceFiles/VulkanPipeline.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/CpuComputeBackend.h`,
  `WindowsApp.Core/SourceFiles/CpuComputeBackend.cpp`
- Create: `WindowsApp.Core/HeaderFiles/BlockMatchAlignKernel.h`,
  `WindowsApp.Core/SourceFiles/BlockMatchAlignKernel.cpp`
- Create: `WindowsApp.Core/HeaderFiles/RobustMergeKernel.h`,
  `WindowsApp.Core/SourceFiles/RobustMergeKernel.cpp`

**Interfaces:**
- `ComputeTypes.h` gains `ComputeResult::NOT_SUPPORTED = 6` and a plain POD
  `struct TileOffset { int dx = 0; int dy = 0; };`.
- `IComputeBackend.h` gains two pure-virtual methods:
  ```cpp
  // Non-overlapping tileSize x tileSize grid (edge tiles clipped to
  // width/height). For each tile, brute-force integer-pixel SAD search in
  // `srcData` within [-searchRadius, +searchRadius] of the tile's position
  // in `refData`, picking the minimum-SAD offset. outOffsets is
  // caller-allocated, tilesX*tilesY entries (tilesX = ceil(width/tileSize),
  // same for Y), row-major. Both buffers RGB48, same width/height.
  virtual ComputeResult BlockMatchAlign(
      const unsigned short* refData, const unsigned short* srcData,
      int width, int height, int tileSize, int searchRadius,
      TileOffset* outOffsets, int tilesX, int tilesY) = 0;

  // Reference-frame-relative Gaussian-weighted merge. frames[0] is the
  // reference (implicit zero offset, weight 1.0, must NOT have an entry in
  // perFrameOffsets); frames[1..numFrames) are aligned via their
  // BlockMatchAlign-computed per-tile offset (perFrameOffsets[k-1],
  // tilesX*tilesY entries each), bilinear-sampled, and out-of-bounds
  // samples excluded from that pixel's weighted average (never contribute
  // a spurious zero - the same "gap-vs-black" lesson RenderExecutor's
  // CombineIgnoringGaps already encodes). sigma: fixed Gaussian-weight
  // noise parameter (see this plan's Architecture note - not per-ISO
  // calibrated). output: caller-allocated width*height*3.
  virtual ComputeResult RobustMergeAccumulate(
      const unsigned short* const* frames, int numFrames,
      const TileOffset* const* perFrameOffsets,
      int width, int height, int tileSize, int tilesX, int tilesY,
      float sigma, unsigned short* output) = 0;
  ```
- `CudaPipeline`/`VulkanPipeline` both implement these as one-line
  `SetError("BlockMatchAlign not implemented on this backend (tracked gap,
  see docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md Task
  2)"); return ComputeResult::NOT_SUPPORTED;` (and same for
  `RobustMergeAccumulate`) — real overrides, not left unimplemented (both
  classes are concrete, not abstract; the pure-virtual methods must be
  overridden to compile at all).
- `CpuComputeBackend`'s overrides delegate to
  `Kernels::BlockMatchAlign`/`Kernels::RobustMergeAccumulate` in the two new
  kernel headers (plain functions, no `Scalar`/`Avx2`/`Avx512` namespacing
  yet — see this plan's Architecture note on why).

- [ ] **Step 1: `ComputeTypes.h` additions** (`NOT_SUPPORTED`, `TileOffset`).
- [ ] **Step 2: `IComputeBackend.h`** — add the two pure-virtual
  declarations with the doc comments above.
- [ ] **Step 3: `BlockMatchAlignKernel.{h,cpp}`** — implement the brute-force
  SAD search per tile. SAD sums `|refPixel - srcPixel|` across all 3 RGB48
  channels over the tile's pixels (clipped at image bounds for edge tiles
  and for candidate windows that would read outside `srcData`'s bounds —
  those candidate offsets are simply not considered, not treated as
  zero-cost). Ties broken by preferring the smallest `|dx|+|dy|` (closest to
  zero motion) — deterministic, and the natural tie-break for "no evidence
  of motion beats an arbitrary equal-cost alternative."
- [ ] **Step 4: `RobustMergeKernel.{h,cpp}`** — implement the weighted merge
  per this plan's Architecture note. For a given output pixel, `frames[0]`
  always contributes (weight 1.0, no bounds check needed since offset is
  implicitly zero and merge only ever runs over the reference's own
  dimensions); `frames[k]` (k≥1) contributes only if
  `(x - offset.dx, y - offset.dy)` bilinear-samples inside `[0,width)×[0,height)`.
  If every non-reference frame is excluded for a pixel, the output is just
  the reference's own value at that pixel (weight-1 average of one sample)
  — never a black/zero pixel, matching the "gap ≠ black" principle.
- [ ] **Step 5: Wire `CpuComputeBackend`** — declare+define the two override
  methods, delegating to the kernels.
- [ ] **Step 6: Wire `CudaPipeline`/`VulkanPipeline`** stubs as described
  above.
- [ ] **Step 7: `WindowsApp.Core.vcxproj`/`CMakeLists.txt`, and
  `WindowsApp.Compute.vcxproj`/`WindowsApp.Vulkan.vcxproj`** — add the two
  new header/source files to both build systems (MSBuild item groups +
  CMake `target_sources`), matching how every prior new file in this repo
  was wired into both.

---

### Task 3: `ProjectManager` burst task seeding

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
- Produces:
  ```cpp
  // One BURST_ALIGN/"frame" task per input image (unit_key = image id,
  // same convention as SeedIngestTasks). Call after every frame has been
  // added via AddInputImage - idempotent like every other Seed*Tasks
  // method (CreateTasksIfAbsent's UNIQUE constraint).
  bool SeedBurstAlignTasks();

  // Exactly one BURST_MERGE/"output" task (unit_key = "merged") and one
  // BURST_FINISH/"output" task (unit_key = "final") - both safe to seed
  // upfront alongside BURST_ALIGN's tasks (see this plan's Architecture
  // note on why burst tasks don't need STAGE3_RENDER's post-hoc seeding
  // treatment).
  bool SeedBurstMergeTasks();
  ```

- [ ] **Step 1: Implement `SeedBurstAlignTasks`** — mirrors
  `SeedIngestTasks`'s existing shape (one task per row in
  `GetInputImages()`, `unit_kind="frame"`).
- [ ] **Step 2: Implement `SeedBurstMergeTasks`** — two single-row
  `CreateTasksIfAbsent` calls (`BURST_MERGE`/"output"/"merged" and
  `BURST_FINISH`/"output"/"final").
- [ ] **Step 3: Header/source consistency check** —
  `grep -n "SeedBurstAlignTasks\|SeedBurstMergeTasks"
  WindowsApp.Core/HeaderFiles/ProjectManager.h
  WindowsApp.Core/SourceFiles/ProjectManager.cpp` — declarations + matching
  definitions.

---

### Task 4: `BurstAlignExecutor`

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/BurstAlignExecutor.h`,
  `WindowsApp.Core/SourceFiles/BurstAlignExecutor.cpp`

**Interfaces:**
- `class BurstAlignExecutor : public ITaskExecutor` — constructor takes
  `ProjectManager&`, `StorageEngine&`, `shared_ptr<IComputeBackend>`,
  `shared_ptr<IImageCodec>` (same shape as `RawIngestExecutor`/
  `AlignExecutor`).
- `Execute`: decode this task's own frame via `DecodeInputImage` (Task 1),
  write it via `WritePixelBuffer("raw_rgb48")`, set `task.outputBlobId`.
  If this frame is the reference (lowest `id` in `GetInputImages()` — frame
  0), that's the entire task: no alignment against itself. Otherwise, ALSO
  decode the reference frame (redundant per-task decode — acceptable for
  MFNR's typically-small burst counts, not the hundreds of chunks the
  panorama path optimizes for; flagged in §9 as a real inefficiency, not
  hidden) and call `BlockMatchAlign(reference, thisFrame, ...)`
  (`tileSize`/`searchRadius` constants, e.g. 16/8 — picked for "a real,
  testable default," not tuned against real photos yet), serializing the
  resulting `TileOffset` array to a small JSON array in `task.checkpointJson`
  (a deliberate repurposing of that field beyond its original "bundle-
  adjustment checkpoint" use — it's just "small opaque per-task state," and
  an offset field for a modest tile grid is tiny; document this repurposing
  in the field's own comment in `Types.h`).

- [ ] **Step 1: Implement `Execute`** per the above.
- [ ] **Step 2: `WindowsApp.Core.vcxproj`/`CMakeLists.txt`** wiring.
- [ ] **Step 3: `Types.h`** — update `Task::checkpointJson`'s comment to
  note the second use (burst-align tile-offset field), not just
  bundle-adjustment.

---

### Task 5: `BurstMergeExecutor` + `BurstFinishExecutor`

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/BurstMergeExecutor.h`,
  `WindowsApp.Core/SourceFiles/BurstMergeExecutor.cpp`

**Interfaces:**
- `class BurstMergeExecutor : public ITaskExecutor` — dispatches by
  `task.stage` (`BURST_MERGE` → real merge, `BURST_FINISH` → passthrough),
  a **composite executor** registered for both stages (same pattern
  `AlignExecutor` already uses for dispatching by `unit_kind` — here it's by
  `stage` instead, since one class instance can be registered against
  multiple `PipelineStage` keys in `TaskScheduler`'s map).
- `ExecuteMerge`: reads every `BURST_ALIGN` task via
  `GetTasksForStage(PipelineStage::BURST_ALIGN)`, `ReadPixelBuffer`s each
  frame's blob, deserializes each non-reference frame's `TileOffset` array
  from its `checkpointJson`, calls `RobustMergeAccumulate`, writes the
  result via `WritePixelBuffer`.
- `ExecuteFinish`: reads `BURST_MERGE`'s single output blob
  (`GetTasksForStage(PipelineStage::BURST_MERGE)[0].outputBlobId`),
  `ReadBlob`s the raw bytes (not `ReadPixelBuffer` — passthrough copies the
  exact bytes, doesn't need to round-trip through `PixelBuffer`), and
  `WriteBlob`s them again as this task's own output — a real, minimal
  "finish" (§9 tracks real finish-stage work as follow-up), not a
  fabricated success. Only proceeds if the `BurstMode` is `MFNR` — other
  modes return `false` with a clear "not yet implemented for this
  BurstMode" `task.errorMessage`, not a silent fake success (`OptimizeExecutor`'s
  existing "no data → valid default, not a failure" precedent doesn't apply
  here, since an unimplemented mode is a genuine failure to surface, not a
  legitimately-empty case).

- [ ] **Step 1: Implement `ExecuteMerge`**.
- [ ] **Step 2: Implement `ExecuteFinish`**.
- [ ] **Step 3: `WindowsApp.Core.vcxproj`/`CMakeLists.txt`** wiring.

---

### Task 6: End-to-end quality-gated test

**Files:**
- Create: `tests/pipeline_e2e_burst/main.cpp`,
  `tests/pipeline_e2e_burst/CMakeLists.txt`
- Modify: root `CMakeLists.txt` (register the new test directory)

**Interfaces:**
- Consumes: `ProjectManager::CreateBurstProject`/`AddInputImage`/
  `SeedBurstAlignTasks`/`SeedBurstMergeTasks`, `PipelineDriver` with the
  three new real executors registered against `BURST_ALIGN`/`BURST_MERGE`/
  `BURST_FINISH`, `CpuComputeBackend`, `JpegCodec`.

- [ ] **Step 1: Kernel-level `BlockMatchAlign` check** — a synthetic
  textured (irregular, not periodic — `tests/pipeline_e2e`'s own fixture
  script already documents why: periodic textures alias block/feature
  matching) reference image, a target built by shifting it a known
  (dx, dy) within the search radius; assert every tile's recovered offset
  matches within the algorithm's stated integer-pixel precision.
- [ ] **Step 2: Kernel-level `RobustMergeAccumulate` check** — a clean
  synthetic scene, N noisy+shifted copies (Gaussian pixel noise, known
  per-tile shifts fed directly as `TileOffset`s rather than routed through
  `BlockMatchAlign` here, to isolate the merge kernel's own correctness
  from alignment's); assert the merged output's PSNR vs. the clean ground
  truth exceeds every individual noisy input's own PSNR vs. ground truth
  (the direct, quantitative proof merging actually reduces noise — not
  just "didn't crash").
- [ ] **Step 3: Full pipeline end-to-end** — generate a clean synthetic
  scene, produce a burst of frames (shifted + noise, encoded to JPEG bytes
  via `JpegCodec::Encode`, written to temp `.jpg` files, added via
  `AddInputImage(path, identityHomography, CfaType::STANDARD_RGB)`), run
  the real `BurstAlignExecutor`/`BurstMergeExecutor` (registered for both
  `BURST_MERGE`/`BURST_FINISH`) through `PipelineDriver::Run` on a
  `CpuComputeBackend`, read `BURST_FINISH`'s output blob back, compute PSNR
  vs. the clean scene, assert it exceeds a threshold set from the actual
  achieved value on a correct run (same data-driven-threshold discipline
  `tests/pipeline_e2e`'s panorama-quality scenario already established —
  not a guessed number).
- [ ] **Step 4: Build + run** —
  ```bash
  cmake --build build --target pipeline_e2e_burst_tests
  ctest --test-dir build -R pipeline_e2e_burst --output-on-failure
  ```

---

### Task 7: Full regression pass

- [ ] **Step 1: Full Linux CMake/ctest suite** — every existing target plus
  the new `pipeline_e2e_burst_tests`, all passing.
- [ ] **Step 2: Windows** — via `windows-host`/`win-thanh`: CMake build+ctest
  (real CUDA+Vulkan hardware, per
  `docs/superpowers/plans/2026-07-21-burst-pipeline-foundation.md`'s
  established verification pattern) and the full `WindowsApp.slnx` MSBuild
  + `WindowsApp.Tests` (`vstest.console.exe`) — confirms the
  `NOT_SUPPORTED` CUDA/Vulkan stubs compile and link cleanly (they must, to
  keep those classes concrete) even though nothing exercises them yet.

## Self-Review

- Spec coverage: `BlockMatchAlign`/`RobustMergeAccumulate` (§3), CPU-first
  with a typed tracked gap for CUDA/Vulkan (§4), and a real quality-gated
  end-to-end MFNR run (§7) — all of Phase 1's roadmap scope.
- Placeholder scan: no placeholder steps; `BURST_FINISH`'s passthrough and
  the `NOT_SUPPORTED` stubs are documented-real minimal implementations,
  not placeholders standing in for something unfinished.
- Type consistency: `TileOffset`/`ComputeResult::NOT_SUPPORTED`/
  `BlockMatchAlign`/`RobustMergeAccumulate` names match across
  `ComputeTypes.h`, `IComputeBackend.h`, both concrete backends, and the
  new kernel/executor files.

## 9. Open scope cuts / follow-up work (carried forward, not hidden)

- **No Gaussian pyramid** in `BlockMatchAlign` — misses motion beyond the
  fixed search radius. A real limitation for handheld bursts with
  significant hand-shake; the panorama path's global feature-match/RANSAC
  step could pre-stabilize frames before block-match runs, reducing how
  much this matters in practice, but that composition isn't wired yet.
- **No sub-pixel refinement** — integer-pixel accuracy only, unlike HDR+'s
  closed-form quadratic fit or Wronski's Lucas-Kanade refinement. Directly
  limits how much noise reduction is achievable (misaligned-by-a-fraction-
  pixel contributions blur, not just fail to help).
- **`sigma` is a fixed constant, not a calibrated per-ISO noise model** —
  HDR+'s real `σ²=Ax+B` Poisson+Gaussian model (§2.1) is real follow-up
  work, not implemented here.
- **`CudaPipeline`/`VulkanPipeline` both `NOT_SUPPORTED`** for both new ops
  — MFNR only runs on the CPU backend until a GPU implementation lands;
  tracked here explicitly per this plan's own Architecture note and
  `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §4's backend-coverage-matrix
  recommendation (this is that matrix's first real entry).
- **Redundant reference-frame decode per `BURST_ALIGN` task** — acceptable
  at MFNR's typical burst sizes (a handful of frames), would need a
  decode-once cache for larger bursts.
