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
