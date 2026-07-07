#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace WindowsApp { namespace Compute { namespace Kernels
{
    // =====================================================================
    // sRGB (16-bit) <-> CIE LAB (D65 white point) - standard textbook
    // formulas (sRGB -> linear -> XYZ -> LAB and back), not this
    // codebase's invention.
    // =====================================================================
    __global__ void RgbToLabKernel(
        const unsigned short* __restrict__ rgb, float* __restrict__ lab, int numPixels);

    __global__ void LabToRgbKernel(
        const float* __restrict__ lab, unsigned short* __restrict__ rgb, int numPixels);

    // =====================================================================
    // Per-channel mean/sum-of-squares reduction (simple atomicAdd-based
    // single pass into double accumulators - not the most efficient
    // approach, a proper tree reduction is follow-up work; host side
    // derives mean/stddev from these sums).
    // =====================================================================
    __global__ void LabStatsKernel(
        const float* __restrict__ lab, int numPixels,
        double* __restrict__ outSum, double* __restrict__ outSumSq);

    // =====================================================================
    // Reinhard color transfer: lab' = (lab - srcMean) * (refStd/srcStd) + refMean
    // =====================================================================
    __global__ void ReinhardTransferKernel(
        float* __restrict__ lab, int numPixels,
        const double* __restrict__ srcMean, const double* __restrict__ srcStd,
        const double* __restrict__ refMean, const double* __restrict__ refStd);
}}}
