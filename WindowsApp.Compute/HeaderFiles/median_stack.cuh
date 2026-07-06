#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace WindowsApp::Compute::Kernels
{
    // =====================================================================
    // Kernel 1: Perspective Warp (backward mapping)
    // =====================================================================
    // Each thread processes one output pixel.
    // Uses inverse homography to find source coordinates.
    // Nearest-neighbor interpolation for speed; bilinear optional.
    __global__ void WarpPerspectiveKernel(
        const unsigned short* __restrict__ srcData,
        unsigned short* __restrict__ dstData,
        int srcW, int srcH, int dstW, int dstH,
        const float* __restrict__ invH,
        int offsetX, int offsetY);

    // =====================================================================
    // Kernel 2: Median Stack with Sigma-Clipping
    // =====================================================================
    // Each thread processes one pixel channel across all inputs.
    // Uses Batcher's Odd-Even Mergesort for fixed-size sorting networks.
    // Sigma-clipping removes outliers before computing median.
    __global__ void MedianStackKernel(
        const unsigned short* __restrict__ inputs,  // flattened: numInputs * numPixels * 3
        unsigned short* __restrict__ output,
        int numInputs, int numPixels,
        float sigmaThreshold);

    // =====================================================================
    // Kernel 3: Gaussian Blur (separable, horizontal pass)
    // =====================================================================
    // Uses shared memory to coalesce reads.
    __global__ void GaussianBlurHorizontal(
        const unsigned short* __restrict__ input,
        unsigned short* __restrict__ output,
        int width, int height,
        const float* __restrict__ kernel, int kernelRadius);

    // =====================================================================
    // Kernel 3: Gaussian Blur (separable, vertical pass)
    // =====================================================================
    __global__ void GaussianBlurVertical(
        const unsigned short* __restrict__ input,
        unsigned short* __restrict__ output,
        int width, int height,
        const float* __restrict__ kernel, int kernelRadius);

    // =====================================================================
    // Kernel 3: Build Laplacian Pyramid (subtract upsampled from current)
    // =====================================================================
    __global__ void LaplacianSubtract(
        const unsigned short* __restrict__ currentLevel,
        const unsigned short* __restrict__ upsampledLower,
        unsigned short* __restrict__ laplacianOut,
        int width, int height);

    // =====================================================================
    // Kernel 3: Blend Laplacian Levels (weighted sum)
    // =====================================================================
    __global__ void BlendLaplacianLevels(
        const unsigned short* __restrict__ lapA,
        const unsigned short* __restrict__ lapB,
        unsigned short* __restrict__ blended,
        int width, int height, float weightA, float weightB);

    // =====================================================================
    // Kernel 3: Reconstruct from Laplacian Pyramid
    // =====================================================================
    __global__ void LaplacianReconstruct(
        const unsigned short* __restrict__ laplacian,
        const unsigned short* __restrict__ upsampledLower,
        unsigned short* __restrict__ output,
        int width, int height);

    // =====================================================================
    // Kernel 4: Gain Compensation
    // =====================================================================
    __global__ void ApplyGainKernel(
        unsigned short* __restrict__ data,
        int numValues, float gain);

    // =====================================================================
    // Utility: Downsample 2x (for pyramid building)
    // =====================================================================
    __global__ void Downsample2x(
        const unsigned short* __restrict__ input,
        unsigned short* __restrict__ output,
        int srcW, int srcH, int dstW, int dstH);

    // =====================================================================
    // Utility: Upsample 2x (for pyramid reconstruction)
    // =====================================================================
    __global__ void Upsample2x(
        const unsigned short* __restrict__ input,
        unsigned short* __restrict__ output,
        int srcW, int srcH, int dstW, int dstH);
}
