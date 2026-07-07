#pragma once
#include "Types.h"
#include "CudaPipeline.h"
#include <vector>
#include <utility>

namespace WindowsApp::Core
{
    struct RansacResult
    {
        Homography homography;
        int inlierCount = 0;
        bool success = false;
    };

    // Reprojection error (Euclidean, pixels) of applying `h` to `src` and
    // comparing against `dst`. Factored out as a free function - pure
    // arithmetic, no GPU - specifically so it's testable without CUDA,
    // unlike RunRansacHomography's orchestration as a whole (which needs
    // Compute::CudaPipeline::TensorEstimateHomography, GPU-only).
    float ReprojectionError(const Homography& h, const Compute::FeaturePoint& src, const Compute::FeaturePoint& dst);

    // Fixed-iteration host-orchestrated RANSAC (docs/superpowers/plans/
    // 2026-07-07-align-stage.md Task 5's Global Constraints - full
    // CUDA-graph-captured RANSAC is later work). Samples 4 random
    // correspondences per iteration, calls
    // CudaPipeline::TensorEstimateHomography, scores inliers on the CPU,
    // keeps the best-scoring result over `iterations` passes.
    RansacResult RunRansacHomography(
        Compute::CudaPipeline& cudaPipeline,
        const std::vector<std::pair<Compute::FeaturePoint, Compute::FeaturePoint>>& correspondences,
        int iterations = 500, float inlierThresholdPx = 3.0f);
}
