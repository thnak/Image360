#pragma once

#include "CudaPipeline.h" // FeaturePoint / BriefDescriptor (plain C++ types)
#include <cuda_runtime.h>
#include <cstdint>

namespace WindowsApp { namespace Compute { namespace Kernels
{
    // =====================================================================
    // Grayscale conversion (FAST/BRIEF both operate on single-channel
    // intensity, not the interleaved RGB nvJPEG decode produces)
    // =====================================================================
    __global__ void RgbToGrayKernel(
        const unsigned char* __restrict__ rgb,
        unsigned char* __restrict__ gray,
        int width, int height);

    // =====================================================================
    // FAST-9 corner detection
    // =====================================================================
    // One thread per candidate pixel. Atomically appends to outPoints
    // (capped at maxPoints via outCount, which may exceed maxPoints - the
    // host side clamps before reading back). No non-maximum suppression
    // in this first pass (documented simplification, not a hidden gap -
    // see docs/superpowers/plans/2026-07-07-align-stage.md Task 3).
    __global__ void FastDetectKernel(
        const unsigned char* __restrict__ grayImage, int width, int height,
        FeaturePoint* __restrict__ outPoints, int* __restrict__ outCount, int maxPoints,
        unsigned char threshold);

    // =====================================================================
    // BRIEF binary descriptor (256-bit)
    // =====================================================================
    // Fixed, deterministic pixel-pair sampling pattern (not any particular
    // published BRIEF table - see docs/superpowers/plans/2026-07-07-align-stage.md
    // Task 3's own note) within a 31x31 patch around each feature point.
    __global__ void BriefDescribeKernel(
        const unsigned char* __restrict__ grayImage, int width, int height,
        const FeaturePoint* __restrict__ points, int numPoints,
        BriefDescriptor* __restrict__ outDescriptors);
}}}
