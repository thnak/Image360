#pragma once

#include <algorithm>

namespace WindowsApp::Core::Kernels
{
    namespace Detail
    {
        // Shared scalar finish step: given a destination pixel's already-
        // computed source-space integer origin (sx0,sy0) and fractional
        // offsets (fx,fy) - the tiered kernels compute these via SIMD
        // across a batch of lanes - bilinear-sample the 4 neighboring
        // source texels and write the blended result. `valid` is false for
        // destination pixels that fell outside the source bounds or hit a
        // near-singular homography (writes 0,0,0, matching the GPU kernel).
        inline void SampleBilinear(
            const unsigned short* srcData, int srcW, int srcH,
            unsigned short* dstPixel, int sx0, int sy0, float fx, float fy, bool valid)
        {
            if (!valid)
            {
                dstPixel[0] = dstPixel[1] = dstPixel[2] = 0;
                return;
            }

            int sx1 = sx0 + 1;
            int sy1 = sy0 + 1;
            float w00 = (1.0f - fx) * (1.0f - fy);
            float w10 = fx * (1.0f - fy);
            float w01 = (1.0f - fx) * fy;
            float w11 = fx * fy;

            for (int c = 0; c < 3; ++c)
            {
                float v00 = static_cast<float>(srcData[(static_cast<size_t>(sy0) * srcW + sx0) * 3 + c]);
                float v10 = static_cast<float>(srcData[(static_cast<size_t>(sy0) * srcW + sx1) * 3 + c]);
                float v01 = static_cast<float>(srcData[(static_cast<size_t>(sy1) * srcW + sx0) * 3 + c]);
                float v11 = static_cast<float>(srcData[(static_cast<size_t>(sy1) * srcW + sx1) * 3 + c]);

                float val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                dstPixel[c] = static_cast<unsigned short>((std::min)((std::max)(val, 0.0f), 65535.0f));
            }
        }
    }

    // Perspective warp (backward mapping via inverse homography, bilinear
    // sample) - matches
    // WindowsApp::Compute::Kernels::WarpPerspectiveKernel's algorithm
    // exactly: out-of-bounds or near-singular destination pixels write
    // (0,0,0). srcData/dstData: RGB48 (unsigned short per channel).
    // homography: 3x3 row-major (9 floats) mapping DESTINATION -> SOURCE
    // (an inverse homography, despite the parameter name matching the
    // public API's "homography" - consistent with the shipped GPU kernel).
    //
    // Note: the Avx2/Avx512 tiers may differ from Scalar by +/-1 in the
    // final 16-bit pixel value on a small fraction of pixels (verified
    // bounded to 1 ULP) - FMA contraction changes intermediate rounding by
    // up to 1 ULP versus separate multiply+add, which occasionally crosses
    // a truncation boundary in the bilinear blend. This is expected
    // cross-ISA floating-point noise (~0.0015% relative error), not a
    // correctness bug - no different than GPU-vs-CPU parity in any other
    // floating-point image pipeline.
    namespace Scalar
    {
        void WarpPerspective(const unsigned short* srcData, int srcW, int srcH,
                              unsigned short* dstData, int dstW, int dstH,
                              const float* homography, int offsetX, int offsetY);
    }
    namespace Avx2
    {
        void WarpPerspective(const unsigned short* srcData, int srcW, int srcH,
                              unsigned short* dstData, int dstW, int dstH,
                              const float* homography, int offsetX, int offsetY);
    }
    namespace Avx512
    {
        void WarpPerspective(const unsigned short* srcData, int srcW, int srcH,
                              unsigned short* dstData, int dstW, int dstH,
                              const float* homography, int offsetX, int offsetY);
    }
}
