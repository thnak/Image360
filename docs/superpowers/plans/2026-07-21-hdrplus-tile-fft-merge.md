# HDR+: Tile FFT Merge + Exposure-Fusion Finish (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 2 of `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §8 — HDR+'s own
merge algorithm. Implements `TileFftMerge` (batched small 2D real FFT +
complex-domain Wiener shrinkage — a genuinely new kernel primitive class per
§2.1/§3, NOT an extension of Phase 1's `RobustMergeAccumulate`) and a real
(non-passthrough) `BURST_FINISH` for `BurstMode::HDR_PLUS`: exposure-fusion
tone mapping via a new small Laplacian-pyramid two-image blend kernel.
Reuses `BlockMatchAlign` and the entire `BurstAlignExecutor`/`Task`/
`TaskScheduler`/`PipelineDriver` plumbing from Phase 1 unchanged — HDR+'s
alignment need is identical to MFNR's (§2.3: alignment is the one thing all
four burst modes share), only the merge and finish stages differ.

**Depends on:** Phase 1 (`docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md`,
commit 5e3784f) — `BlockMatchAlign`, `BurstAlignExecutor`,
`BurstMergeExecutor`'s dispatch-by-`task.stage` shape, `BurstCommon.h`'s
shared tile-grid constants.

**Architecture:**
- **`TileFftMerge`**: per-tile (non-overlapping `kBurstTileSize` grid, same
  grid `BlockMatchAlign` already computes offsets for — see §9 for the
  overlap-window cut this implies), per-RGB-channel (post-demosaic, not
  pre-demosaic raw Bayer planes — see §9), 2D real→complex FFT of the
  reference tile and of each aligned alternate-frame tile, then per-
  frequency-bin Wiener shrinkage of each alternate's *difference* from the
  reference: `A(ω) = |D(ω)|² / (|D(ω)|² + c·σ²)`, `D(ω) = altSpectrum(ω) −
  refSpectrum(ω)`, merged spectrum = `refSpectrum + mean_k(A_k(ω)·D_k(ω))`.
  Inverse FFT reconstructs the merged tile; edge tiles are clamp-to-edge
  sampled up to the fixed `tileSize` (never zero-padded — a hard edge-to-
  zero transition would ring badly through the FFT). This is the same
  per-frequency-bin shrinkage `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §2.1
  describes, just without HDR+'s calibrated per-ISO `σ²=Ax+B` noise model
  (§9) and without the paper's raised-cosine 50%-tile-overlap windowing
  (§9) — both explicit, tracked cuts, same discipline Phase 1 used for its
  own scope cuts.
- **FFT implementation**: a small self-contained iterative radix-2
  Cooley-Tukey complex FFT (`std::complex<float>`), applied row-then-column
  for the 2D transform. Requires `tileSize` to be a power of two — checked
  at runtime (`ComputeResult::INVALID_PARAM` otherwise), satisfied today by
  `kBurstTileSize = 16`. No new third-party dependency; this repo has no
  FFT library vendored and 16×16 tiles are tiny enough that a textbook
  radix-2 implementation is both correct and fast enough without needing
  FFTW/pocketfft.
- **`FuseTwoExposures`** (new `Kernels::ExposureFusion`, NOT a reuse of
  `Kernels::SeamBlend::BlendChunkContributors`): a self-contained N=2-image
  Laplacian-pyramid blend using the same Burt-Adelson Gaussian/Laplacian
  build+blend+reconstruct math `SeamBlend.cpp` already proved, but without
  any of `SeamBlend`'s ownership/seam-routing/hole-filling machinery — that
  machinery exists specifically for panorama's *partial-coverage* per-
  contributor warped buffers (holes, seams between disagreeing images),
  which exposure fusion doesn't have (both exposures are dense, full-frame,
  same underlying merged content, just different tone curves). Per-pixel
  blend weight is Mertens-style well-exposedness only (no contrast/
  saturation terms — see §9): `w(v) = exp(-((v/65535 − 0.5)²) / (2·0.2²))`
  per channel, product across channels, normalized so the two images' weight
  masks sum to 1 at every pixel.
- **`BurstMergeExecutor::Execute`'s mode gate widens** from "only MFNR" to
  "MFNR or HDR_PLUS" — `ExecuteMerge`/`ExecuteFinish` each switch on
  `BurstMode` internally (MFNR path unchanged, HDR_PLUS path new). Any
  other mode still returns the existing "not yet implemented" failure.
- **HDR_PLUS's `BURST_FINISH`** reads `BURST_MERGE`'s output buffer,
  synthesizes two tone-curve exposures from it (`v' = 65535·(v/65535)^γ`,
  γ<1 brightens/lifts shadows, γ>1 darkens/compresses highlights — fixed
  gammas, not scene-adaptive metering, §9), fuses them via
  `FuseTwoExposures`, writes the result. A real transform, not a
  passthrough — HDR+ needs tone mapping (unlike MFNR, which legitimately
  didn't), per `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §2.1's Finish stage.
- **No `ProjectManager` changes** — `CreateBurstProject`/
  `SeedBurstAlignTasks`/`SeedBurstMergeTasks` already take `BurstMode` as
  opaque data and don't branch on it; HDR+ projects reuse them unchanged,
  confirming Phase 0's "second pipeline family" design generalizes past
  just MFNR without modification.

**Tech Stack:** C++20, existing `WindowsApp.Core` structure; extends
`tests/pipeline_e2e_burst` (same executable/target as Phase 1 — shares its
`GenerateCleanScene16`/`ShiftAndNoise16`/`Psnr16`/`NarrowTo8` helpers rather
than duplicating them in a new target) with HDR+-specific scenarios.

## Global Constraints

- No CUDA/Vulkan kernel implementation for `TileFftMerge` in this plan —
  `NOT_SUPPORTED` stub only, same tracked-gap discipline as Phase 1's two
  ops (§9).
- No overlap-add tile windowing, no calibrated per-ISO noise model, no
  scene-adaptive exposure-fusion metering — all explicit, tracked scope
  cuts (§9), not silently pretended complete.
- Don't touch MFNR's existing `ExecuteMerge`/`ExecuteFinish` behavior —
  Phase 1's `pipeline_e2e_burst` scenarios (Steps 1-3) must keep passing
  unmodified; that's the regression check.

---

### Task 1: `IComputeBackend::TileFftMerge`

**Files:**
- Modify: `WindowsApp.Compute/HeaderFiles/IComputeBackend.h`
- Modify: `WindowsApp.Compute/HeaderFiles/CudaPipeline.h`,
  `WindowsApp.Compute/SourceFiles/CudaPipeline.cpp`
- Modify: `WindowsApp.Vulkan/HeaderFiles/VulkanPipeline.h`,
  `WindowsApp.Vulkan/SourceFiles/VulkanPipeline.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/CpuComputeBackend.h`,
  `WindowsApp.Core/SourceFiles/CpuComputeBackend.cpp`
- Create: `WindowsApp.Core/HeaderFiles/TileFftMergeKernel.h`,
  `WindowsApp.Core/SourceFiles/TileFftMergeKernel.cpp`

**Interfaces:**
```cpp
// HDR+'s merge (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.1) - per-tile,
// per-frequency-bin Wiener shrinkage of each alternate frame's spectral
// DIFFERENCE from the reference, NOT MFNR's spatial-domain
// RobustMergeAccumulate (a different algorithm, per SS2.3's correction).
// Same tile grid/offset convention as RobustMergeAccumulate: frames[0] is
// the reference (no perFrameOffsets entry); frames[1..numFrames) aligned
// via perFrameOffsets[k-1]. tileSize MUST be a power of two (the FFT
// implementation is radix-2) - INVALID_PARAM otherwise. noiseVariance is
// the Wiener shrinkage's c*sigma^2 term (fixed, not per-ISO calibrated -
// see the plan doc's SS9). output: caller-allocated width*height*3.
// CPU-only - CUDA/Vulkan return NOT_SUPPORTED.
virtual ComputeResult TileFftMerge(
    const unsigned short* const* frames, int numFrames,
    const TileOffset* const* perFrameOffsets,
    int width, int height, int tileSize, int tilesX, int tilesY,
    float noiseVariance, unsigned short* output) = 0;
```

- [ ] **Step 1: `IComputeBackend.h`** — add the pure-virtual declaration
  with the doc comment above.
- [ ] **Step 2: `TileFftMergeKernel.{h,cpp}`** — radix-2 `Fft1D`
  (iterative, bit-reversal permutation + butterfly passes, forward/inverse
  via a `bool inverse` flag and `1/n` inverse normalization) and `Fft2D`
  (row passes then column passes, in place on an `n*n` `std::complex<float>`
  buffer). `TileFftMerge` itself: for each tile, for each channel, FFT the
  reference tile, then for each alternate frame FFT its (offset-sampled,
  clamp-to-edge) tile, accumulate the Wiener-shrunk difference into the
  merged spectrum, inverse-FFT, write the valid (non-clamped) sub-rectangle
  into `output`. Guard the Wiener division (`power / (power + noiseVariance)`)
  against a zero denominator (returns 0 contribution, not NaN — matches
  Phase 1's own `std::normal_distribution(stddev=0)` NaN lesson: never let a
  degenerate-input division produce NaN silently).
- [ ] **Step 3: Wire `CpuComputeBackend`** — declare+define the override,
  validating `tileSize` is a power of two before delegating to the kernel.
- [ ] **Step 4: Wire `CudaPipeline`/`VulkanPipeline`** stubs (same
  `SetError(...); return ComputeResult::NOT_SUPPORTED;` shape as Phase 1's
  two ops).
- [ ] **Step 5: Build wiring** — `WindowsApp.Core.vcxproj`/`CMakeLists.txt`
  (new kernel files), `WindowsApp.Compute.vcxproj`/`WindowsApp.Vulkan.vcxproj`
  (interface-only change, no new files there).

---

### Task 2: `Kernels::ExposureFusion::FuseTwoExposures`

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/ExposureFusionKernel.h`,
  `WindowsApp.Core/SourceFiles/ExposureFusionKernel.cpp`

**Interfaces:**
```cpp
namespace WindowsApp::Core::Kernels::ExposureFusion
{
    // Mertens-style two-image Laplacian-pyramid blend, well-exposedness-
    // weighted only (no contrast/saturation terms - see the plan doc's
    // SS9). low/high: RGB48, width*height*3, same dimensions - typically
    // two synthetic tone-curve renderings of the SAME underlying image
    // (see BurstMergeExecutor's HDR_PLUS finish path), not independently
    // captured brackets, so no ghosting/alignment concern here (unlike
    // real multi-capture bracket fusion). outResult: resized internally.
    void FuseTwoExposures(
        const unsigned short* low, const unsigned short* high,
        int width, int height, int numBands,
        std::vector<unsigned short>& outResult);
}
```

- [ ] **Step 1: Implement well-exposedness weight computation** — per
  pixel, per image, `w = product over channels of exp(-((v/65535-0.5)^2)/(2*0.2^2))`;
  normalize `wLow/(wLow+wHigh)`, `wHigh/(wLow+wHigh)` per pixel (guard the
  `wLow+wHigh ≈ 0` case — both images equally badly exposed at that pixel —
  by falling back to 0.5/0.5, not a divide-by-zero).
- [ ] **Step 2: Implement the pyramid blend** — Gaussian pyramid of each
  weight mask, Laplacian pyramid of each exposure image (plain, no hole-
  filling needed — both inputs are dense), per-level weighted sum
  `wLowPyr*lapLow + wHighPyr*lapHigh`, top-down reconstruction. Reuse the
  same blur/downsample/upsample math `SeamBlendKernels.h`'s anonymous-
  namespace helpers already implement, reimplemented locally here rather
  than exported from `SeamBlend.cpp` (keeps `SeamBlend.cpp`'s panorama-
  specific internals private; this is a small enough amount of duplicated
  math — 3 helper functions — that a shared-header extraction isn't worth
  the coupling between panorama-blend and burst-blend internals).
- [ ] **Step 3: Build wiring** — `WindowsApp.Core.vcxproj`/`CMakeLists.txt`.

---

### Task 3: Wire `TileFftMerge`/`FuseTwoExposures` into `BurstMergeExecutor`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/BurstMergeExecutor.h` (doc comment
  only)
- Modify: `WindowsApp.Core/SourceFiles/BurstMergeExecutor.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/BurstCommon.h` (new HDR+ constants)

**Interfaces:**
- `BurstCommon.h` gains `kHdrPlusNoiseVariance` (Wiener shrinkage's `c·σ²`,
  fixed constant, same "real testable default, not calibrated" status as
  `kBurstMergeSigma`) and `kHdrPlusBrightGamma`/`kHdrPlusDarkGamma` (the two
  synthetic-exposure tone curves' γ values).
- `BurstMergeExecutor::Execute`'s top-level mode gate becomes `mode ==
  BurstMode::MFNR || mode == BurstMode::HDR_PLUS` (was MFNR-only).
- `ExecuteMerge` switches on `m_projectManager.GetBurstMode()`: MFNR path
  unchanged (calls `RobustMergeAccumulate`); HDR_PLUS path calls
  `TileFftMerge` instead, same frame/offset assembly logic (shared, not
  duplicated — factor the "gather reference + non-reference frames/offsets"
  block into a small local helper used by both branches).
- `ExecuteFinish` switches on `GetBurstMode()`: MFNR path unchanged
  (passthrough); HDR_PLUS path reads `BURST_MERGE`'s output via
  `ReadPixelBuffer` (not `ReadBlob` — needs real pixel data to tone-map,
  unlike MFNR's byte-level passthrough), builds the two synthetic exposures,
  calls `FuseTwoExposures`, writes the result via `WritePixelBuffer`.

- [ ] **Step 1: Add HDR+ constants to `BurstCommon.h`**.
- [ ] **Step 2: Refactor `ExecuteMerge`'s frame/offset gathering** into a
  shared local helper (both MFNR and HDR_PLUS need the identical "collect
  reference + non-reference buffers/offsets from completed `BURST_ALIGN`
  tasks" logic) — a pure refactor of Phase 1's existing code, then branch on
  `BurstMode` to call `RobustMergeAccumulate` or `TileFftMerge`.
- [ ] **Step 3: Implement HDR_PLUS's `ExecuteFinish` branch** — synthetic
  exposure generation + `FuseTwoExposures` call, as described above.
- [ ] **Step 4: Widen the mode gate** in `Execute`.
- [ ] **Step 5: Regression check** — Phase 1's existing MFNR scenarios
  (`pipeline_e2e_burst_tests` Steps 1-3) must still pass unmodified.

---

### Task 4: End-to-end HDR+ tests (extends `tests/pipeline_e2e_burst`)

**Files:**
- Modify: `tests/pipeline_e2e_burst/main.cpp`

**Interfaces:**
- Reuses `GenerateCleanScene16`/`ShiftAndNoise16`/`NarrowTo8`/`Psnr16`
  from Phase 1's existing helpers in the same file.

- [ ] **Step 1: Kernel-level `TileFftMerge` check** (Step 4 in the test's
  own numbering, continuing after Phase 1's 3 steps) — same shape as Phase
  1's `RunRobustMergeKernelCheck`: clean scene, N noisy+shifted copies,
  known offsets fed directly, assert merged PSNR vs. clean exceeds the
  reference frame's own PSNR.
- [ ] **Step 2: Kernel-level `FuseTwoExposures` check** (Step 5) — a
  synthetic HDR scene with a genuinely dark region and a genuinely bright
  region (values spanning close to 0 and close to 65535, not just mid-
  tones), two synthetic exposures via the two fixed gamma curves, fused
  result. Assert the fused output tracks the *brightened* exposure in the
  dark region (closer to it than to the darkened exposure there) and the
  *darkened/compressed* exposure in the bright region (closer to it than to
  the brightened exposure there) — the direct, quantitative proof
  well-exposedness weighting is actually picking the better-exposed source
  per region, not just "didn't crash" or "returned something the right
  size."
- [ ] **Step 3: Full pipeline end-to-end** (Step 6) — same burst-generation
  shape as Phase 1's `RunFullPipelineQualityScenario` but
  `BurstMode::HDR_PLUS`. Quality-gate the **`BURST_MERGE`** stage's output
  (not `BURST_FINISH`'s) against the clean scene via PSNR — `BURST_FINISH`
  deliberately changes pixel values via tone mapping, so a PSNR-vs-linear-
  ground-truth comparison doesn't apply post-tone-map (unlike MFNR, whose
  passthrough finish made the whole-pipeline PSNR check valid end to end).
  Separately assert `BURST_FINISH` completes with a correctly-sized output
  that is NOT byte-identical to `BURST_MERGE`'s output (proof the tone-map
  transform actually ran, not a hidden passthrough).
- [ ] **Step 4: Build + run** —
  ```bash
  cmake --build build --target pipeline_e2e_burst_tests
  ctest --test-dir build -R pipeline_e2e_burst --output-on-failure
  ```

---

### Task 5: Full regression pass

- [ ] **Step 1: Full Linux CMake/ctest suite** — every existing target,
  including Phase 1's now-unmodified MFNR scenarios and the new HDR+
  scenarios, all passing.
- [ ] **Step 2: Windows** — via `windows-host`/`win-thanh`: CMake build+
  ctest (real CUDA+Vulkan hardware), full `WindowsApp.slnx` MSBuild +
  `WindowsApp.Tests` (`vstest.console.exe`) — confirms the new
  `TileFftMerge` `NOT_SUPPORTED` CUDA/Vulkan stubs compile/link cleanly.

## Self-Review

- Spec coverage: `TileFftMerge` (§2.1/§3's "genuinely new kernel primitive
  class"), CPU-first with a typed tracked gap for CUDA/Vulkan (§4), a real
  (non-passthrough) HDR+ finish stage via exposure fusion, and quality-gated
  tests for both new kernels plus the full pipeline — all of Phase 2's
  roadmap scope (§8).
- Placeholder scan: no placeholder steps; the `NOT_SUPPORTED` stubs are the
  same documented-real minimal implementation pattern Phase 1 already
  established, not new placeholders.
- Type consistency: `TileFftMerge`/`FuseTwoExposures` names and signatures
  match across `IComputeBackend.h`, `CpuComputeBackend`, the two new kernel
  files, `BurstMergeExecutor`, and the test file.

## 9. Open scope cuts / follow-up work (carried forward, not hidden)

- **No 50%-overlap raised-cosine tile windowing** — HDR+'s real merge uses
  overlapped, windowed tiles specifically to avoid blocking artifacts at
  tile boundaries (`docs/COMPUTATIONAL_PHOTOGRAPHY.md` §2.1). This MVP
  reuses the same non-overlapping grid `BlockMatchAlign` already computes
  offsets for, which is simpler to wire but can show visible blocking at
  tile edges on real photos with high-frequency content there. Tracked
  follow-up, not attempted here.
- **`noiseVariance` is a fixed constant, not HDR+'s calibrated per-ISO
  `σ²=Ax+B` Poisson+Gaussian model** — identical scope cut to Phase 1's
  `sigma`, same reasoning.
- **Post-demosaic RGB merge, not pre-demosaic Bayer-plane merge** — the
  real HDR+ algorithm merges independent raw Bayer color planes before
  demosaic (§2.1); this MVP merges after demosaic (matching how
  `BurstAlignExecutor`/`RobustMergeAccumulate` already work for MFNR in
  this codebase), which is simpler but merges already-interpolated data
  instead of the raw sensor signal HDR+'s noise model assumes.
- **`FuseTwoExposures` is well-exposedness-only**, no Mertens' contrast
  (Laplacian-magnitude) or saturation (per-pixel channel stddev) terms —
  real scope cut vs. the full Mertens 2007 weighting; well-exposedness
  alone is enough to demonstrably favor the better-exposed source per
  region (Task 4's test proves this) but a full implementation would blend
  more intelligently in ambiguous regions.
- **Synthetic exposure gammas are fixed constants, not scene-adaptive
  metering** — real HDR+/Night Sight derive exposure/tone parameters from
  scene content (§2.2's motion-metering is a related but distinct concept);
  this MVP always applies the same two γ curves regardless of the actual
  merged image's dynamic range.
- **`CudaPipeline`/`VulkanPipeline` both `NOT_SUPPORTED`** for
  `TileFftMerge` — HDR+ only runs on the CPU backend until a GPU
  implementation lands, same tracked-gap discipline as Phase 1's two ops.
- **No dehaze/chromatic-aberration/unsharp-mask/dithering** — HDR+'s real
  13-step Finish (§2.1) is much more than tone mapping; this phase
  implements only the exposure-fusion tone-mapping step, the piece the
  roadmap doc explicitly calls out as directly reusing existing pyramid-
  blend machinery.
