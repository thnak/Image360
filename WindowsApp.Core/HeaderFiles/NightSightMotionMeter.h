#pragma once
#include "ComputeTypes.h"
#include <vector>

namespace WindowsApp::Core::Kernels
{
    // docs/superpowers/plans/2026-07-22-night-sight.md - a batch-pipeline
    // reinterpretation of docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.2's
    // pre-capture "motion metering": rather than picking burst
    // length/exposure before a shutter this engine never controls, meters
    // the *actual* per-frame motion BurstAlignExecutor already measured
    // (BlockMatchAlign + RefineOffsetsSubPixel) to adaptively drop
    // unusably-shifted frames and scale StructureTensorKernelRegression's
    // robustness weighting.
    struct MotionMeteringResult
    {
        // Parallel to the input perFrameOffsets vector (one entry per
        // non-reference frame) - true if the frame should be kept.
        std::vector<bool> keepFrame;
        // Adaptive noiseVariance for StructureTensorKernelRegression -
        // derived from baseNoiseVariance and the kept frames' aggregate
        // motion, not baseNoiseVariance itself.
        float noiseVariance = 0.0f;
    };

    // perFrameOffsets[i] is non-reference frame i's per-tile TileOffsetF
    // field (same layout BurstMergeExecutor::GatherAlignedFrames already
    // deserializes). baseNoiseVariance is kNightSightBaseNoiseVariance
    // (production) or a test's own injected noise scale, matching every
    // other merge kernel's "test tunes its own scale" precedent.
    MotionMeteringResult MeterMotion(
        const std::vector<std::vector<Compute::TileOffsetF>>& perFrameOffsets,
        float baseNoiseVariance);
}
