#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace WindowsApp { namespace Compute { namespace Kernels
{
    // =====================================================================
    // Demosaic Kernel 1: Black Level Subtract
    // =====================================================================
    // In-place, saturating subtract on the raw CFA plane (one sample per
    // pixel, not yet demosaiced into RGB).
    __global__ void BlackLevelSubtractKernel(
        unsigned short* __restrict__ cfaData,
        int numPixels, unsigned short blackLevel);

    // =====================================================================
    // Demosaic Kernel 2: White Balance
    // =====================================================================
    // In-place, per-channel gain from LibRaw's cam_mul[4]. Channel index
    // per pixel is derived from `filters` the same way LibRaw's own
    // COLOR(row,col) macro does, so this kernel doesn't need LibRaw
    // headers pulled into WindowsApp.Compute.
    __global__ void WhiteBalanceKernel(
        unsigned short* __restrict__ cfaData,
        int width, int height,
        float camMul0, float camMul1, float camMul2, float camMul3,
        uint32_t filters);

    // =====================================================================
    // Demosaic Kernel 3: Bayer Demosaic (bilinear)
    // =====================================================================
    // One thread per output pixel - standard bilinear Bayer interpolation
    // (docs/ARCHITECTURE.md SS4.1: bilinear for v1, Malvar-He-Cutler/AHD
    // is a later upgrade that doesn't change the surrounding kernel chain).
    __global__ void DemosaicBayerKernel(
        const unsigned short* __restrict__ cfaData,
        unsigned short* __restrict__ rgbOut,
        int width, int height, uint32_t filters);

    // =====================================================================
    // Demosaic Kernel 4: Color Matrix (camera RGB -> sRGB)
    // =====================================================================
    // One thread per pixel: out = rgbCam (3x4 homogeneous) * [inR, inG, inB, 1].
    __global__ void ColorMatrixKernel(
        const unsigned short* __restrict__ rgbIn,
        unsigned short* __restrict__ rgbOut,
        int numPixels,
        const float* __restrict__ rgbCam); // 12 floats, row-major 3x4

    // =====================================================================
    // Demosaic Kernel 5: Tone Curve (linear passthrough for v1)
    // =====================================================================
    // A real kernel, not skipped - docs/ARCHITECTURE.md SS4.1 explicitly
    // scopes the tone curve to linear passthrough for v1, but the graph
    // shape is still five real stages so a later plan can swap this one
    // kernel's body without touching anything else.
    __global__ void ToneCurveKernel(
        const unsigned short* __restrict__ rgbIn,
        unsigned short* __restrict__ rgbOut,
        int numPixels);
}}}
