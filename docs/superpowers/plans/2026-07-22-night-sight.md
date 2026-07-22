# Night Sight (Phase 3 fast-follow)

## Goal

Implement Night Sight (`docs/COMPUTATIONAL_PHOTOGRAPHY.md` §2.2), the last
GCam-class merge mode in the §8 phase table before the CLI (Phase 4) /AI
façade (Phase 5) phases. Per §2.2 and §8: Night Sight is **not** its own
merge algorithm — it rides on Phase 3's `StructureTensorKernelRegression`
(same kernel Super Res Zoom uses) plus two new pieces: a motion-metering
step and a distinct "painterly" tone-curve finish.

## Depends on

Phase 3 (commit `d9bb854`) — `RefineOffsetsSubPixel`, `TileOffsetF`,
`IComputeBackend::StructureTensorKernelRegression`, and the
`BurstAlignExecutor`/`BurstMergeExecutor` SUPER_RES wiring, all reused
unchanged. `BurstMode::NIGHT_SIGHT` and its `.vfp` string serialization
already exist since Phase 0 — no schema work needed here.

## Scope reinterpretation (read this before the code)

§2.2 describes "motion metering" as a **pre-capture** stage: run optical
flow on live viewfinder frames to pick burst length (6 frames tripod, up to
15 handheld) and per-frame exposure (up to 333ms handheld, 1s tripod)
*before* the shutter fires. This engine has no live viewfinder — it is a
batch post-processor over an already-fixed, already-captured set of input
images (RAW or JPEG). Picking "how many frames to capture" or "what
exposure to use" is not a decision this codebase is in a position to make.

Rather than build a fictitious pre-capture-simulation stage that doesn't
fit the engine's actual capture model, motion metering is reinterpreted
here as **post-hoc adaptive frame selection + adaptive merge robustness**,
computed from data the pipeline already has:

- `BurstAlignExecutor` already runs `BlockMatchAlign` +
  `RefineOffsetsSubPixel` per frame (identical to `SUPER_RES` — Night
  Sight needs the same sub-pixel offsets `StructureTensorKernelRegression`
  requires). That per-tile offset field IS a measured motion field between
  each frame and the reference — literally the same information a
  viewfinder optical-flow pass would produce, just measured from the real
  burst instead of a preview stream.
- `BurstMergeExecutor` (for `NIGHT_SIGHT` only) runs a new function,
  `Kernels::MeterMotion`, over all frames' offset fields to (a) exclude any
  frame whose own motion is a severe outlier relative to the rest of the
  burst (mirrors "the flow estimate rejects an unusable frame" — a real
  photographic use case, e.g. someone bumped the camera mid-burst) and (b)
  scale `StructureTensorKernelRegression`'s `noiseVariance` — smaller
  (tighter agreement required, more conservative/ghost-safe) under high
  aggregate motion, larger (more permissive, more denoising) under low
  motion — mirroring §2.2's real handheld-vs-tripod behavior split without
  needing a literal tripod-detection heuristic.

This is a genuine, testable, real piece of new orchestration logic — not a
literal new `PipelineStage`/`ITaskExecutor`. A brand-new `PipelineStage`
was considered and rejected: `BURST_ALIGN` already decodes+aligns every
frame (there is no earlier point in the burst pipeline where a metering
pass could run without redundantly re-decoding every RAW/JPEG first), so
the natural, cheapest, least-duplicative home for this logic is exactly
where the alignment data already lives — consumed by `BurstMergeExecutor`
between `GatherAlignedFrames` and the `StructureTensorKernelRegression`
call. `PipelineDriver`'s stage sequence, `TaskScheduler`, and the `.vfp`
schema are all untouched by this phase.

Same scope discipline as prior phases: the auto-white-balance classifier
and astrophotography mode (§2.2's last two bullets) are explicitly **not**
built — out of scope, no learned-inference path exists yet (that's Phase
5), and astro mode is "a further extension of the same pipeline," not a
new design concern.

## Architecture

1. **`NightSightMotionMeter.h`/`.cpp`** (new, `WindowsApp::Core::Kernels`,
   host-side — pure per-frame arithmetic over an already-computed offset
   field, same "doesn't need `IComputeBackend`" rationale as
   `RefineOffsetsSubPixel`/`HomographyMath`):
   `MeterMotion(perFrameOffsets, baseNoiseVariance) -> MotionMeteringResult
   {keepFrame[], noiseVariance}`. Per non-reference frame, computes mean
   offset magnitude across tiles; a frame is excluded if its own mean
   magnitude exceeds both a relative threshold (vs. the burst's median) and
   an absolute floor tied to `kBurstSearchRadius` (near-search-radius
   offsets are a `BlockMatchAlign` reliability red flag on their own,
   independent of the rest of the burst — see `BlockMatchAlignKernel.h`).
   Never excludes so many frames that fewer than 2 remain (reference +
   1) — a burst that's *entirely* high motion should still merge
   something, not degrade to a single frame. `noiseVariance` scales
   `baseNoiseVariance` by a factor derived from the (post-exclusion) mean
   motion, clamped to a bounded range so a pathological input can't zero it
   out or blow it up.
2. **`PainterlyToneCurveKernel.h`/`.cpp`** (new,
   `WindowsApp::Core::Kernels::PainterlyToneCurve`): `Apply(src, width,
   height, shadowGamma, highlightRolloff, vignetteStrength, outDst)` — per
   channel, per pixel: shadow-crushing power curve (`gamma > 1`) composed
   with a soft Reinhard-style highlight rolloff (avoids clipping, keeps the
   flat "painterly" look §2.2 describes), then a multiplicative radial
   vignette (darkened surrounds) based on normalized distance from image
   center. A distinct, independent implementation from HDR+'s
   `ExposureFusion` (different algorithm family — single-image tone curve,
   not a multi-exposure Laplacian-pyramid blend — same "genuinely
   different op" precedent as `TileFftMerge` vs.
   `StructureTensorKernelRegression`).
3. **`BurstMergeExecutor::ExecuteMerge`**: widen the existing
   `mode == BurstMode::SUPER_RES` branch's condition to also cover
   `NIGHT_SIGHT` (same `StructureTensorKernelRegression` call, same
   sub-pixel-offset gathering path), but:
   - `scaleFactor` is `kNightSightScaleFactor = 1` for `NIGHT_SIGHT` (no
     upsampling — Night Sight denoises/merges at native resolution, it
     isn't zooming), vs. `kSuperResScaleFactor` for `SUPER_RES`.
   - For `NIGHT_SIGHT` only, `MeterMotion` runs first over
     `gathered.perFrameOffsetsF`; excluded frames are dropped from both
     `framePtrs`/`offsetPtrsF` before the call (reference frame is never
     eligible for exclusion), and its returned `noiseVariance` (not the
     raw `kNightSightBaseNoiseVariance` constant) is passed to
     `StructureTensorKernelRegression`.
4. **`BurstAlignExecutor`**: widen the existing
   `GetBurstMode() == BurstMode::SUPER_RES` sub-pixel-refinement branch to
   also cover `NIGHT_SIGHT` (same reason `NIGHT_SIGHT`'s merge needs
   `TileOffsetF` input — no new logic, just an `||` on the mode check).
   `GatherAlignedFrames`'s `useSubPixelOffsets` gets the same `||`.
5. **`BurstMergeExecutor::ExecuteFinish`**: add a real (non-passthrough)
   `NIGHT_SIGHT` branch — reads `BURST_MERGE`'s output, runs
   `PainterlyToneCurve::Apply`, writes the result as `BURST_FINISH`'s
   output. Same shape as the existing `HDR_PLUS` branch (`ApplyToneCurve`
   + `FuseTwoExposures`), just a different (single-image) tone operator.
6. **`BurstCommon.h`** gains: `kNightSightScaleFactor = 1`,
   `kNightSightBaseNoiseVariance` (same "typical camera ISO, not
   calibrated per-ISO" scope cut as the other three merge-tuning
   constants), `kNightSightMotionExclusionRelativeFactor`,
   `kNightSightMotionExclusionAbsolutePx`,
   `kNightSightMinKeptNonReferenceFrames`,
   `kNightSightNoiseVarianceMinScale`/`MaxScale`/`Gain` (motion-metering
   tuning), and `kNightSightShadowGamma`/`HighlightRolloff`/
   `VignetteStrength` (painterly tone-curve tuning) — all documented as
   real-but-untuned defaults, matching every prior phase's constants.

`ProjectManager` needs **zero** changes — `BurstMode::NIGHT_SIGHT`,
`CreateBurstProject`, `GetBurstMode`, and the burst task-seeding functions
already handle it identically to the other three modes (verified against
`ProjectManager.cpp`'s existing exhaustive `BurstMode` string mapping and
`tests/engine_smoke`'s coverage before starting this phase).

## Tech stack

Same as Phase 3: header-only-declared, `.cpp`-implemented free functions
under `WindowsApp::Core::Kernels`, C++20, no new third-party dependency.
CPU-only for this phase (both new pieces are host-side orchestration/tone
math, not `IComputeBackend` ops — no CUDA/Vulkan stub needed since neither
`MeterMotion` nor `PainterlyToneCurve` are `IComputeBackend` members).

## Global constraints

- No behavior change for `MFNR`/`HDR_PLUS`/`SUPER_RES` — every touched
  function's existing branches for those modes must be provably unchanged
  (same "widen with `||`, don't restructure" discipline as Phase 2/3).
- `docs/COMPUTATIONAL_PHOTOGRAPHY.md`'s "tests are not optional" bar
  applies: kernel-level tests for both new pieces, plus a full-pipeline
  PSNR-gated `NIGHT_SIGHT` scenario in `tests/pipeline_e2e_burst`.
- Same dual-platform verification discipline (Linux CMake/ctest, then real
  Windows/CUDA/Vulkan hardware via win-thanh: CMake+ctest, full
  `WindowsApp.slnx` MSBuild, `WindowsApp.Tests`) before calling this done.

## Tasks

1. `NightSightMotionMeter.h`/`.cpp` + kernel-level test (low-motion burst:
   all frames kept, larger noiseVariance; one severe-outlier frame: that
   frame excluded, others kept).
2. `PainterlyToneCurveKernel.h`/`.cpp` + kernel-level test (shadows darken
   more than a plain gamma=1 mapping; corners darken relative to center on
   a uniform-gray input).
3. `BurstAlignExecutor`/`BurstMergeExecutor::GatherAlignedFrames` mode-check
   widening (`SUPER_RES` → `SUPER_RES || NIGHT_SIGHT`).
4. `BurstMergeExecutor::ExecuteMerge` `NIGHT_SIGHT` branch (scaleFactor=1,
   `MeterMotion`-filtered frames/adaptive noiseVariance).
5. `BurstMergeExecutor::ExecuteFinish` `NIGHT_SIGHT` branch
   (`PainterlyToneCurve::Apply`).
6. `BurstCommon.h` constants.
7. CMake + `.vcxproj` wiring for the two new source files.
8. Full `NIGHT_SIGHT` pipeline scenario in `tests/pipeline_e2e_burst`
   (Step 12): PSNR gate on `BURST_MERGE` vs. reference-frame-alone, using
   Step 3/6's *strict* "must beat reference" bar (not Step 9's relaxed
   floor) — verified empirically to clear a real ~2.3 dB margin, since
   Night Sight's native-resolution merge (scaleFactor=1) isn't exposed to
   Super Res Zoom's upsampling-amplified sub-pixel-precision sensitivity;
   `BURST_FINISH` checked structurally (dimensions match, corners
   measurably darker than center, not byte-identical to `BURST_MERGE`).
9. Dual-platform verification (Linux ctest, then win-thanh: CMake+ctest,
   MSBuild, WindowsApp.Tests).

## Self-review

- Does this reuse Phase 3's merge kernel unchanged, per §2.2's explicit
  instruction ("it does not get its own merge op")? Yes —
  `StructureTensorKernelRegression` itself is untouched; only its call
  site's `scaleFactor`/`noiseVariance` arguments vary by mode.
- Does `MeterMotion` risk silently discarding a legitimate burst down to
  1 frame? No — `kNightSightMinKeptNonReferenceFrames` floor.
- Is the "no live viewfinder" scope cut disclosed, not silently
  papered over? Yes — see "Scope reinterpretation" above; recorded again
  in `project_image360_computational_photography_progress.md` once
  verified.

## 9. Open scope cuts

- Auto-white-balance classifier (learned) — deferred to Phase 5 (AI
  façade), no learned-inference path exists yet.
- Astrophotography mode — an extension of this same pipeline per §2.2,
  not a new design concern; not built.
- `MeterMotion`'s thresholds are heuristic, not calibrated against real
  handheld/tripod burst statistics (same "real but untuned default" scope
  cut as every other burst-mode constant in `BurstCommon.h`).
- Phase 3's known tile-scatter limitation (small-tile single-level LK
  variance under noise, `SubPixelRefineKernel.cpp`) is inherited as-is by
  `NIGHT_SIGHT` since it reuses the same alignment+merge kernels — not
  re-litigated or re-fixed in this phase.
