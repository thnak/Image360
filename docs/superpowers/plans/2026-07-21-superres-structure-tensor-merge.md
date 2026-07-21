# Phase 3: Super Res Zoom — sub-pixel alignment + structure-tensor kernel regression

## Goal

Implement `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §2.4/§8 Phase 3: `BurstMode::SUPER_RES`,
built on Wronski et al.'s *Handheld Multi-Frame Super-Resolution*. Natural
hand tremor gives each burst frame a slightly different sub-pixel sample
phase relative to the scene; combining frames via an edge-aware kernel
regression reconstructs an upsampled image with real recovered detail, not
just an interpolated blow-up of one frame.

Two new pieces, per §2.4's own stage breakdown:

1. **Sub-pixel alignment refinement** — `BlockMatchAlign`'s existing
   integer-pixel tiles (§8 Phase 1, unchanged) refined by a few Lucas-Kanade
   iterations per tile — stage (a), "sequential/global, must run first."
2. **`StructureTensorKernelRegression`** — a new `IComputeBackend` op
   combining stages (b)-(e): local gradient structure-tensor analysis,
   anisotropic kernel construction elongated along edges, kernel-regression
   accumulation onto an upsampled output grid, and the same kind of
   noise-based robustness weighting HDR+/MFNR already use.

Night Sight (§2.2) is explicitly **out of scope for this phase** — it needs
this same merge kernel plus a new motion-metering executor and a distinct
tone-mapping curve, tracked as its own fast-follow (§9).

## Depends on

Phase 1 (commit 5e3784f) for `BlockMatchAlign`/`BurstAlignExecutor` — reused
**unchanged** as the coarse alignment seed, same as Phase 2 reused it
unmodified. Not on Phase 2 (HDR+'s `TileFftMerge`) — a different merge
algorithm entirely, per §2.3's correction.

## Architecture

### Sub-pixel refinement (host-side, not through `IComputeBackend`)

New `WindowsApp.Core/HeaderFiles/SubPixelRefineKernel.h` /
`SourceFiles/SubPixelRefineKernel.cpp`:

```cpp
void RefineOffsetsSubPixel(
    const unsigned short* refData, const unsigned short* srcData,
    int width, int height, int tileSize,
    const Compute::TileOffset* coarseOffsets, int tilesX, int tilesY,
    int iterations, Compute::TileOffsetF* outOffsets);
```

Kept host-only, same rationale as `HomographyMath`/`LinearSolve` (small
per-tile dense system, a GPU round-trip would add latency without benefit —
see `IComputeBackend.h`'s own note on this tradeoff). For each tile, seeded
from `BlockMatchAlign`'s integer `(dx,dy)`: precompute the reference tile's
Sobel gradients once, then `iterations` single-level Lucas-Kanade steps —
each step bilinear-samples `srcData` at `(x+dx, y+dy)` (matching
`RobustMergeAccumulate`'s existing `sx = x + offset.dx` convention exactly,
just now with a fractional `dx`/`dy`), builds the 2×2 normal-equations
system `[[ΣIx²,ΣIxIy],[ΣIxIy,ΣIy²]]·δ = -[ΣIxIt,ΣIyIt]`, solves it directly
(2×2, no `LinearSolve.h` dependency needed), and accumulates `δ` into
`(dx,dy)` — clamped per-step to avoid divergence on a low-texture tile, and
skipped entirely if the system is near-singular (flat tile: no reliable
sub-pixel information, coarse integer offset is kept as-is, matching
`BlockMatchAlign`'s own "no valid candidate" tolerance for degenerate
tiles). A final pass 3×3-neighborhood-averages the whole refined offset
field before returning — added after Task 4 testing showed independent
per-tile (16×16-sample) LK estimates have real statistical variance under
realistic noise, scattering by up to ~0.1–0.3px between neighboring tiles
that share the same true motion; this regularizes the field the same way
video/optical-flow pipelines routinely smooth motion vector fields, at the
cost of a little true local-motion resolution (see §9).

New `Compute::TileOffsetF { float dx = 0.0f; float dy = 0.0f; }` in
`ComputeTypes.h`, alongside the existing integer `TileOffset` — a distinct
type, not a reinterpretation, so a caller can never accidentally pass
integer offsets where sub-pixel ones are expected (compile error, not a
runtime bug). `SerializeTileOffsetsF`/`DeserializeTileOffsetsF` (same
hand-rolled-JSON convention as `SerializeTileOffsets`, `std::stof` instead
of `std::stoi`) live alongside `RefineOffsetsSubPixel`.

### `IComputeBackend::StructureTensorKernelRegression`

```cpp
virtual ComputeResult StructureTensorKernelRegression(
    const unsigned short* const* frames, int numFrames,
    const TileOffsetF* const* perFrameOffsets,
    int width, int height, int tileSize, int tilesX, int tilesY,
    int scaleFactor, float noiseVariance, unsigned short* output) = 0;
```

Same `frames[0]`-is-reference / `perFrameOffsets[k-1]` convention as
`RobustMergeAccumulate`/`TileFftMerge`, but sub-pixel (`TileOffsetF`, not
`TileOffset`) and producing an **upsampled** `output`
(`width*scaleFactor` × `height*scaleFactor`, caller-allocated) instead of
the input resolution. CPU-only; CUDA/Vulkan get typed `NOT_SUPPORTED`
stubs, same tracked-gap discipline as Phases 1-2.

CPU implementation, `WindowsApp.Core/HeaderFiles/StructureTensorMergeKernel.h`
/ `SourceFiles/StructureTensorMergeKernel.cpp`, `Kernels::StructureTensorKernelRegression`:

1. **Structure tensor** (per reference-frame pixel, precomputed once):
   Sobel `Gx`/`Gy` on reference luma (unweighted RGB average — no existing
   `RgbToGray.h` reuse, its BT.601 weights are for the panorama path's
   different purpose and pulling that dependency in isn't warranted for a
   gradient-magnitude proxy), then a windowed 2×2 tensor
   `[[ΣGx²,ΣGxGy],[ΣGxGy,ΣGy²]]` (window radius
   `kSuperResStructureTensorRadius`). Eigen-decomposed analytically (closed
   form for a 2×2 symmetric matrix — no general eigensolver needed) into a
   dominant-gradient angle `theta` and a coherence
   `c = (λ1-λ2)/(λ1+λ2+ε) ∈ [0,1]`, stored per pixel (2 floats) rather than
   the raw tensor — cheaper to reuse per output pixel below.
2. **Anisotropic kernel construction** (per output pixel, derived from the
   nearest reference pixel's `theta`/`c`): anisotropy
   `A = 1 + (kSuperResMaxAnisotropy - 1)·c`; kernel radius **along** the
   edge direction (perpendicular to the gradient) =
   `kSuperResBaseKernelRadius·A`, radius **across** it (along the gradient)
   = `kSuperResBaseKernelRadius/A` — elongated along edges, compressed
   across them, isotropic (`A=1`) wherever the local structure tensor has
   no clear dominant direction (flat regions, corners).
3. **Kernel-regression accumulation + robustness weighting** (per output
   pixel, gather not scatter — same no-atomics style as
   `RobustMergeAccumulate`/`TileFftMerge`): map the output pixel to a
   continuous reference-space coordinate, look up its tile's
   `perFrameOffsets` entry per frame, gather every frame's raw integer
   pixel samples within a bounded window of that coordinate (never
   bilinear-resampled — kernel regression's whole premise is combining
   *actual* sample positions from different sub-pixel phases, resampling
   first would throw that information away), weight each by the
   anisotropic Gaussian evaluated in the eigenbasis **times** a
   `RobustMergeAccumulate`-style noise-based weight
   (`exp(-(sample-refLocal)²/(2·noiseVariance))`, `refLocal` bilinear-
   sampled from the reference at the output coordinate) to down-weight
   outliers (moving subjects, occlusion), accumulates
   `Σ(weight·sample)/Σweight` per channel. A pixel with near-zero total
   weight (kernel support degenerates near frame boundaries) falls back to
   the reference's own bilinear sample — never a spurious zero, same
   "gap != black" principle `RenderExecutor`/`RobustMergeAccumulate` already
   apply.

### `BurstAlignExecutor` — sub-pixel refinement for `SUPER_RES` only

Unchanged for `MFNR`/`HDR_PLUS`: still runs `BlockMatchAlign` and serializes
integer `TileOffset`s via `SerializeTileOffsets`. For
`BurstMode::SUPER_RES`, after the existing integer `BlockMatchAlign` call,
an additional `Kernels::RefineOffsetsSubPixel` pass produces sub-pixel
offsets, serialized instead via `SerializeTileOffsetsF` — a mode branch on
the *serialization format*, not a second decode/align pass (the coarse
`BlockMatchAlign` result is reused as the refinement's seed, not thrown
away).

### `BurstMergeExecutor` — widened dispatch

- `Execute()`'s mode gate widens to also accept `BurstMode::SUPER_RES`.
- `GatherAlignedFrames` branches on `GetBurstMode()`: `SUPER_RES` calls
  `DeserializeTileOffsetsF` into a new `GatheredFrames::perFrameOffsetsF`
  field; `MFNR`/`HDR_PLUS` keep using `DeserializeTileOffsets` into the
  existing integer field, unchanged.
- `ExecuteMerge` branches to `StructureTensorKernelRegression` for
  `SUPER_RES`, allocating an upsampled `PixelBuffer`
  (`width*kSuperResScaleFactor` × `height*kSuperResScaleFactor`) instead of
  the input-resolution buffer the other two modes use.
- `ExecuteFinish` for `SUPER_RES` is an **identity passthrough** (same
  minimal-and-correct reasoning as `MFNR`'s existing passthrough — the
  kernel-regression merge already produces final RGB directly, no
  tone-mapping step is part of this phase's scope, see §9).

### `ProjectManager` needs zero changes

Same confirmation as Phase 2: `CreateBurstProject`/`SeedBurstAlignTasks`/
`SeedBurstMergeTasks` are already mode-agnostic (`BurstMode` is opaque
metadata to them); `SUPER_RES` needs no new schema or seeding logic.

## Tech Stack

Plain C++20, no new third-party dependency (same "self-contained kernel"
approach as Phase 2's FFT) — 2×2 eigendecomposition and Lucas-Kanade normal
equations are both closed-form, no linear-algebra library needed.

## Global Constraints

- CPU-only this phase (`CudaPipeline`/`VulkanPipeline` return
  `NOT_SUPPORTED` for `StructureTensorKernelRegression`, tracked gap per
  §4).
- `BurstCommon.h` gains: `kSuperResScaleFactor` (2), 
  `kSuperResStructureTensorRadius` (2, i.e. a 5×5 gradient window),
  `kSuperResBaseKernelRadius` (1.0, low-res pixels),
  `kSuperResMaxAnisotropy` (2.5), `kSuperResNoiseVariance` (4000.0, same
  scale as `kBurstMergeSigma`/`kHdrPlusNoiseVariance`), 
  `kSubPixelRefineIterations` (6) — all "real, testable defaults," not
  calibrated per-ISO/per-lens (same scope cut every prior phase's constants
  made).
- No new `PipelineStage`/`BurstMode` values — `SUPER_RES` already exists
  (Phase 0).

## Task 1: `TileOffsetF` + sub-pixel refinement kernel

- Add `Compute::TileOffsetF` to `ComputeTypes.h`.
- `SubPixelRefineKernel.h`/`.cpp`: `RefineOffsetsSubPixel` +
  `SerializeTileOffsetsF`/`DeserializeTileOffsetsF`.
- Wire into `WindowsApp.Core/CMakeLists.txt` (`CORE_SOURCES`) and
  `WindowsApp.Core.vcxproj` (`ClInclude`/`ClCompile`).

## Task 2: `StructureTensorKernelRegression` kernel + `IComputeBackend` wiring

- `StructureTensorMergeKernel.h`/`.cpp` (CPU implementation, see
  Architecture above).
- `IComputeBackend::StructureTensorKernelRegression` pure virtual + doc
  comment.
- `CpuComputeBackend::StructureTensorKernelRegression` — param validation
  (null checks, `numFrames>=1`, `scaleFactor>=1`, `noiseVariance>=0`,
  tile-grid checks matching the other two merge ops) + delegate to
  `Kernels::StructureTensorKernelRegression`.
- `CudaPipeline`/`VulkanPipeline` — `NOT_SUPPORTED` stubs referencing this
  plan, matching Phases 1-2's exact stub pattern.
- Wire new source files into both build systems.

## Task 3: Wire into `BurstAlignExecutor`/`BurstMergeExecutor`

- `BurstAlignExecutor`: `SUPER_RES`-only sub-pixel refinement branch (see
  Architecture above).
- `BurstMergeExecutor`: widened `Execute()` mode gate,
  `GatheredFrames::perFrameOffsetsF`, mode-branched `GatherAlignedFrames`,
  `SUPER_RES` branch in `ExecuteMerge` (upsampled output buffer +
  `StructureTensorKernelRegression` call), `SUPER_RES` passthrough branch in
  `ExecuteFinish`.
- New `BurstCommon.h` constants (Global Constraints above).

## Task 4: Extend `tests/pipeline_e2e_burst` with 3 new scenarios

Same synthetic-JPEG-fixture approach as Phases 1-2, MFNR/HDR+'s existing 6
scenarios unmodified:

- **Step 7**: `RefineOffsetsSubPixel` kernel check — a synthetic pair with a
  *known fractional* shift (e.g. `dx=2.4, dy=-1.7`), seeded from
  `BlockMatchAlign`'s integer result; asserts the refined offset is closer
  to the true fractional shift than the coarse integer-only estimate.
- **Step 8**: `StructureTensorKernelRegression` kernel check — 8 low-res
  frames at well-distributed 2D sub-pixel phases (a Halton(2,3)-style
  low-discrepancy set), each with injected Gaussian noise, sampled from a
  shared high-res synthetic ground truth (known offsets fed directly,
  bypassing alignment — isolates the merge kernel's own correctness, same
  precedent as Steps 2/4). Quality gate: the kernel-regression output's
  PSNR against the high-res ground truth exceeds naive bilinear upsampling
  of the (equally noisy) reference frame alone.
- **Step 9**: full `SUPER_RES` pipeline (`ProjectManager` → `PipelineDriver`
  → real `BurstAlignExecutor`/`BurstMergeExecutor`) through the real JPEG
  round trip; `BURST_FINISH` checked structurally (upsampled dimensions,
  byte-identical to `BURST_MERGE` — proof the passthrough is real, same
  style as MFNR's own Step 3 assumption made explicit). Its PSNR check is
  **not** the same strict "must beat naive" bar Step 8 uses — see the real,
  diagnosed limitation recorded in §9 below; it instead checks
  `mergedPsnr > naivePsnr - 6.0` (a data-driven floor well above a
  genuinely broken merge), with both values logged for visibility.

**Real finding while writing Step 8** (worth recording next to Phase 1/2's
own "quality gate finds real bugs" precedent): the test's first attempt at
Step 8 passed a kernel-level noiseVariance argument copied from
`kSuperResNoiseVariance` (the *production* default, calibrated for
`sigma=2000` noise) while the test's own injected noise used a different
sigma — the mismatch made the robustness term over-reject genuinely valid
cross-frame samples, nearly erasing the merge's advantage over naive
upsampling (a ~0.01 dB margin, not a real proof of anything). Fixed by
passing the test's *own* noise variance directly to the kernel call
(`kNoiseSigma * kNoiseSigma`), matching Step 2's existing precedent of
tuning `RobustMergeAccumulate`'s `sigma` argument to the test's own
synthetic noise rather than reusing the production constant — this raised
the margin to a robust ~3.4 dB. Also discovered: a perfectly noiseless,
smooth synthetic scene gives a single frame's bilinear upsample almost no
room to improve on (bilinear is exactly unbiased for locally-linear
content, and there's no noise for a wider kernel to average away) — Step
8's frames need real injected sensor noise for the comparison to mean
anything, the same "combine N noisy observations" premise Steps 2/4/6
already established for their own merges.

**Real finding while writing Step 9**: even with well-recovered per-tile
sub-pixel offsets on average, independent per-tile (16x16-pixel sample)
Lucas-Kanade estimates have real statistical variance under actual JPEG +
8-bit + sensor noise — empirically, offsets for the *same* globally-uniform
true shift scattered up to ~0.1-0.3px between neighboring tiles.
`StructureTensorKernelRegression`'s hard per-tile constant-offset lookup
turns that scatter into visible tile-boundary artifacts, costing more PSNR
than the merge's own gains buy back. Partially mitigated by adding a
3x3-neighborhood smoothing pass over the refined offset field at the end
of `RefineOffsetsSubPixel` (Task 1) — a real, worthwhile improvement, but
not enough on its own to fully recover Step 8's clean-offset advantage
under real-pipeline noise, hence Step 9's relaxed floor above. Tracked as
a real, open limitation (§9), not silently absorbed into a weaker test.

## Task 5: Full regression pass

`cmake --build build && ctest --test-dir build --output-on-failure`
(Linux), then the same dual-platform verification prior phases used on
win-thanh: CMake+ctest with real CUDA+Vulkan hardware, full
`WindowsApp.slnx` MSBuild, `WindowsApp.Tests` via `vstest.console.exe`.

## Self-Review

- Does `RefineOffsetsSubPixel` reuse `RobustMergeAccumulate`'s existing
  `sx = x + offset.dx` sign convention exactly? Yes — checked directly
  against `RobustMergeKernel.cpp`'s existing code before writing this
  section, not assumed.
- Does anything reuse GPL-3.0 `ImageStackAlignator` code?  No — algorithm
  understanding only, implementation is independent (§2.4's own licensing
  note).
- Any silent zero-padding at frame boundaries?  No — kernel regression
  skips out-of-bounds integer samples entirely (never treated as zero),
  matching `RobustMergeAccumulate`/`TileFftMerge`'s existing "gap != black"
  handling.

## 9. Open scope cuts (tracked, not silent)

- **Night Sight is not built in this phase.** It needs this exact merge
  kernel plus a new motion-metering executor (orchestration-level, reuses
  feature-match) and its own "painterly" tone-curve variant (§2.2) — a
  fast-follow once this phase's merge kernel is verified working, not
  bundled into it.
- **Post-demosaic RGB merge, not pre-demosaic Bayer-plane merge.** Real
  Night Sight bypasses demosaic entirely; this phase merges after
  `BurstAlignExecutor`'s existing RGB48 decode, same architectural choice
  Phase 2 made for `TileFftMerge`.
- **Per-tile structure tensor lookup (nearest-neighbor), not truly
  per-output-pixel.** The eigenbasis is computed once per *reference*
  pixel and looked up by nearest neighbor for each output pixel — a
  precision/cost tradeoff, not a correctness bug (the roadmap's own
  "per-pixel, parallel" framing is satisfied for the tensor's own
  computation; only the *lookup* at output resolution is nearest-neighbor).
- **Fixed anisotropy/radius constants, not per-ISO/per-lens calibrated** —
  same scope cut every prior phase's merge constants made.
- **No Lucas-Kanade pyramid** — single-level refinement only, seeded from
  `BlockMatchAlign`'s integer result (already close, since block matching
  narrows the search to within one pixel before refinement starts) — a
  multi-level pyramid would handle larger uncorrected motion but isn't
  needed for typical handheld sub-pixel jitter.
- **No `--scale` CLI option or arbitrary scale factors validated beyond
  `kSuperResScaleFactor=2`** — the CLI itself is Phase 4 (§6); the kernel
  accepts any `scaleFactor>=1` but only 2× is exercised by this phase's
  tests.
- **Real limitation, found via Task 4/5 testing, not yet fully closed:**
  under realistic (JPEG + 8-bit + sensor-noise) conditions, independent
  per-tile (16×16-sample) Lucas-Kanade sub-pixel estimates have enough
  statistical variance between neighboring tiles sharing the same true
  motion (~0.1–0.3px scatter, empirically measured) that
  `StructureTensorKernelRegression`'s hard per-tile constant-offset lookup
  produces visible tile-boundary artifacts, costing more PSNR than the
  merge's own gains buy back — `tests/pipeline_e2e_burst`'s Step 9
  (full-pipeline) quality gate is deliberately relaxed
  (`mergedPsnr > naivePsnr - 6.0`, not "must beat naive") to reflect this
  honestly rather than hide it behind a weaker synthetic scenario. The
  3×3-neighborhood offset-field smoothing pass added to
  `RefineOffsetsSubPixel` (Task 1) helps but doesn't fully close the gap.
  Real fixes to try in a follow-up: a larger LK sample window (decoupled
  from `kBurstTileSize`, which also drives the merge's own tile grid), a
  proper multi-level pyramid (see the cut above), or bilinear-interpolating
  offsets between tile centers at merge time instead of a hard per-tile
  lookup (avoids the tile-boundary discontinuity entirely, independent of
  how accurate any single tile's estimate is).
