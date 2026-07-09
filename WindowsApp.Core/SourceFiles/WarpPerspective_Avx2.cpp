#include "pch.h"
#include "HeaderFiles/WarpPerspectiveKernels.h"

#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace WindowsApp::Core::Kernels::Avx2
{
    namespace
    {
        constexpr int kLanes = 8;
    }

    // Vectorizes the per-destination-pixel coordinate transform (inverse
    // homography apply, division, floor, bounds check) across 8 lanes at
    // once - the actual 4-texel bilinear gather+blend is inherently
    // scattered memory access, so it stays scalar via the shared
    // Detail::SampleBilinear helper, fed the SIMD-computed coordinates.
    void WarpPerspective(const unsigned short* srcData, int srcW, int srcH,
                         unsigned short* dstData, int dstW, int dstH,
                         const float* invH, int offsetX, int offsetY)
    {
        const __m256 h0 = _mm256_set1_ps(invH[0]);
        const __m256 h3 = _mm256_set1_ps(invH[3]);
        const __m256 h6 = _mm256_set1_ps(invH[6]);
        const __m256 lanesOffset = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256 epsilon = _mm256_set1_ps(1e-10f);
        const __m256 signMask = _mm256_set1_ps(-0.0f);

        for (int y = 0; y < dstH; ++y)
        {
            float dstY = static_cast<float>(y + offsetY);
            float rowY = invH[1] * dstY + invH[2];
            float rowYb = invH[4] * dstY + invH[5];
            float rowYc = invH[7] * dstY + invH[8];
            unsigned short* dstRow = dstData + static_cast<size_t>(y) * dstW * 3;

            int x = 0;
            for (; x + kLanes <= dstW; x += kLanes)
            {
                __m256 dstX = _mm256_add_ps(_mm256_set1_ps(static_cast<float>(x + offsetX)), lanesOffset);

                __m256 srcX = _mm256_add_ps(_mm256_mul_ps(h0, dstX), _mm256_set1_ps(rowY));
                __m256 srcY = _mm256_add_ps(_mm256_mul_ps(h3, dstX), _mm256_set1_ps(rowYb));
                __m256 srcW1 = _mm256_add_ps(_mm256_mul_ps(h6, dstX), _mm256_set1_ps(rowYc));

                __m256 absW1 = _mm256_andnot_ps(signMask, srcW1);
                __m256 singularMask = _mm256_cmp_ps(absW1, epsilon, _CMP_LT_OQ);

                srcX = _mm256_div_ps(srcX, srcW1);
                srcY = _mm256_div_ps(srcY, srcW1);

                __m256 sx0f = _mm256_floor_ps(srcX);
                __m256 sy0f = _mm256_floor_ps(srcY);
                __m256 fxVec = _mm256_sub_ps(srcX, sx0f);
                __m256 fyVec = _mm256_sub_ps(srcY, sy0f);

                alignas(32) int sx0Arr[kLanes], sy0Arr[kLanes];
                alignas(32) float fxArr[kLanes], fyArr[kLanes];
                _mm256_store_si256(reinterpret_cast<__m256i*>(sx0Arr), _mm256_cvttps_epi32(sx0f));
                _mm256_store_si256(reinterpret_cast<__m256i*>(sy0Arr), _mm256_cvttps_epi32(sy0f));
                _mm256_store_ps(fxArr, fxVec);
                _mm256_store_ps(fyArr, fyVec);

                alignas(32) uint32_t singularArr[kLanes];
                _mm256_store_si256(reinterpret_cast<__m256i*>(singularArr), _mm256_castps_si256(singularMask));

                for (int lane = 0; lane < kLanes; ++lane)
                {
                    unsigned short* dstPixel = dstRow + static_cast<size_t>(x + lane) * 3;
                    bool singular = singularArr[lane] != 0;
                    bool valid = !singular && sx0Arr[lane] >= 0 && sy0Arr[lane] >= 0
                                 && sx0Arr[lane] + 1 < srcW && sy0Arr[lane] + 1 < srcH;
                    Detail::SampleBilinear(srcData, srcW, srcH, dstPixel, sx0Arr[lane], sy0Arr[lane], fxArr[lane], fyArr[lane], valid);
                }
            }

            // Scalar tail for the remainder (< kLanes elements).
            for (; x < dstW; ++x)
            {
                float dstX = static_cast<float>(x + offsetX);
                float srcX = invH[0] * dstX + rowY;
                float srcY = invH[3] * dstX + rowYb;
                float srcW1 = invH[6] * dstX + rowYc;

                unsigned short* dstPixel = dstRow + static_cast<size_t>(x) * 3;
                if (std::fabs(srcW1) < 1e-10f)
                {
                    dstPixel[0] = dstPixel[1] = dstPixel[2] = 0;
                    continue;
                }
                srcX /= srcW1;
                srcY /= srcW1;
                int sx0 = static_cast<int>(std::floor(srcX));
                int sy0 = static_cast<int>(std::floor(srcY));
                float fx = srcX - sx0;
                float fy = srcY - sy0;
                bool valid = (sx0 >= 0 && sy0 >= 0 && sx0 + 1 < srcW && sy0 + 1 < srcH);
                Detail::SampleBilinear(srcData, srcW, srcH, dstPixel, sx0, sy0, fx, fy, valid);
            }
        }
    }
}
