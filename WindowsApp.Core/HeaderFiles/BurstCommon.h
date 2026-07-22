#pragma once
#include "ProjectManager.h"
#include <limits>
#include <vector>

namespace WindowsApp::Core
{
    // Shared by BurstAlignExecutor and BurstMergeExecutor - docs/
    // superpowers/plans/2026-07-21-mfnr-block-match-merge.md Tasks 4-5.
    // Kept in one place rather than duplicated per-file so the two
    // executors can never silently drift apart on tile geometry.

    // Picked as "a real, testable default," not tuned against real photos
    // yet.
    inline constexpr int kBurstTileSize = 16;
    inline constexpr int kBurstSearchRadius = 8;

    // RobustMergeAccumulate's Gaussian-weight noise parameter - also not
    // calibrated per-ISO (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.3's
    // correction: that's HDR+'s real noise model, tracked follow-up here).
    inline constexpr float kBurstMergeSigma = 2000.0f;

    // TileFftMerge's Wiener-shrinkage c*sigma^2 term (HDR+ - docs/
    // superpowers/plans/2026-07-21-hdrplus-tile-fft-merge.md) - a real,
    // testable default, not a calibrated per-ISO noise model either (same
    // scope cut as kBurstMergeSigma, tracked in that plan's SS9).
    inline constexpr float kHdrPlusNoiseVariance = 4000.0f;

    // BurstMergeExecutor's HDR_PLUS BURST_FINISH synthesizes two tone-curve
    // exposures from the merged image (v' = 65535*(v/65535)^gamma) and
    // fuses them via Kernels::ExposureFusion::FuseTwoExposures - fixed
    // gammas, not scene-adaptive metering (tracked follow-up, same plan's
    // SS9). kHdrPlusBrightGamma < 1 lifts shadows; kHdrPlusDarkGamma > 1
    // compresses highlights.
    inline constexpr float kHdrPlusBrightGamma = 0.5f;
    inline constexpr float kHdrPlusDarkGamma = 2.0f;

    // Super Res Zoom (docs/superpowers/plans/
    // 2026-07-21-superres-structure-tensor-merge.md) - kSuperResScaleFactor
    // is the only scale factor this phase's tests exercise (the kernel
    // itself accepts any scaleFactor >= 1). kSuperResNoiseVariance is
    // StructureTensorKernelRegression's robustness-weighting denominator
    // (plays the role of sigma^2 in exp(-delta^2/(2*noiseVariance)),
    // unlike kBurstMergeSigma which is passed as sigma itself - kept at
    // the equivalent scale, 2000^2, for the same "typical camera ISO"
    // assumption as kBurstMergeSigma/kHdrPlusNoiseVariance, not a
    // calibrated per-ISO noise model either (same scope cut). Test code
    // that isolates the merge kernel with its own known noise level tunes
    // this argument directly rather than reusing this production default -
    // see tests/pipeline_e2e_burst's Step 8. kSubPixelRefineIterations is
    // RefineOffsetsSubPixel's per-tile Lucas-Kanade iteration count.
    inline constexpr int kSuperResScaleFactor = 2;
    inline constexpr float kSuperResNoiseVariance = 4000000.0f;
    inline constexpr int kSubPixelRefineIterations = 6;

    // Night Sight (docs/superpowers/plans/2026-07-22-night-sight.md) -
    // reuses StructureTensorKernelRegression (same as Super Res Zoom) but
    // at native resolution (scaleFactor=1, no upsampling) with a motion-
    // metered noiseVariance instead of a fixed constant. kNightSight*Noise*
    // is the base value Kernels::MeterMotion scales (same "typical camera
    // ISO, not calibrated per-ISO" scope cut as kSuperResNoiseVariance).
    inline constexpr int kNightSightScaleFactor = 1;
    inline constexpr float kNightSightBaseNoiseVariance = 4000000.0f;

    // MeterMotion's frame-exclusion thresholds: a non-reference frame is
    // dropped if its mean per-tile offset magnitude exceeds the burst's
    // median by this relative factor, OR exceeds this absolute pixel
    // floor outright (near kBurstSearchRadius - BlockMatchAlign's own
    // reliability boundary, independent of the rest of the burst). Never
    // excludes below this many kept non-reference frames.
    inline constexpr float kNightSightMotionExclusionRelativeFactor = 2.5f;
    inline constexpr float kNightSightMotionExclusionAbsolutePx = 6.0f;
    inline constexpr int kNightSightMinKeptNonReferenceFrames = 2;

    // MeterMotion's noiseVariance scale = clamp(1 + gain*(referencePx -
    // meanKeptMotion), minScale, maxScale) - a burst whose aggregate
    // motion is below this reference point (tripod-like, confident
    // alignment) gets a LARGER noiseVariance (more permissive, more
    // denoising); above it (handheld-like, less confident alignment) gets
    // a SMALLER one (tighter agreement required, ghost-safer). Bounded so
    // a pathological input can't zero out or blow up the parameter.
    inline constexpr float kNightSightMotionReferencePx = 3.0f;
    inline constexpr float kNightSightNoiseVarianceGain = 0.08f;
    inline constexpr float kNightSightNoiseVarianceMinScale = 0.4f;
    inline constexpr float kNightSightNoiseVarianceMaxScale = 1.5f;

    // PainterlyToneCurve's BURST_FINISH parameters (docs/
    // COMPUTATIONAL_PHOTOGRAPHY.md SS2.2) - real-but-untuned defaults, same
    // scope cut as kHdrPlusBrightGamma/kHdrPlusDarkGamma. shadowGamma > 1
    // crushes shadows/midtones; highlightRolloff softly compresses
    // highlights; vignetteStrength darkens the image radially outward from
    // center ("darkened surrounds").
    inline constexpr float kNightSightShadowGamma = 1.5f;
    inline constexpr float kNightSightHighlightRolloff = 0.2f;
    inline constexpr float kNightSightVignetteStrength = 0.35f;

    // The lowest-id frame (the first one added) is the alignment/merge
    // reference - not necessarily images.front(), since GetInputImages()'s
    // ordering isn't part of its documented contract.
    inline int BurstReferenceImageId(const std::vector<InputImageModel>& images)
    {
        int minId = (std::numeric_limits<int>::max)();
        for (const auto& img : images) minId = (std::min)(minId, img.id);
        return minId;
    }
}
