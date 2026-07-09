#include "pch.h"
#include "HeaderFiles/WarpPerspectiveKernels.h"

#include <cmath>
#include <cstdint>
#include <immintrin.h>

namespace WindowsApp::Core::Kernels::Avx512
{
    namespace
    {
        constexpr int kLanes = 16;
    }

    // Same approach as the AVX2 tier (see WarpPerspective_Avx2.cpp),
    // widened to 16 lanes per batch.
    void WarpPerspective(const unsigned short* srcData, int srcW, int srcH,
                         unsigned short* dstData, int dstW, int dstH,
                         const float* invH, int offsetX, int offsetY)
    {
        const __m512 h0 = _mm512_set1_ps(invH[0]);
        const __m512 h3 = _mm512_set1_ps(invH[3]);
        const __m512 h6 = _mm512_set1_ps(invH[6]);
        const __m512 lanesOffset = _mm512_set_ps(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        const __m512 epsilon = _mm512_set1_ps(1e-10f);

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
                __m512 dstX = _mm512_add_ps(_mm512_set1_ps(static_cast<float>(x + offsetX)), lanesOffset);

                __m512 srcX = _mm512_add_ps(_mm512_mul_ps(h0, dstX), _mm512_set1_ps(rowY));
                __m512 srcY = _mm512_add_ps(_mm512_mul_ps(h3, dstX), _mm512_set1_ps(rowYb));
                __m512 srcW1 = _mm512_add_ps(_mm512_mul_ps(h6, dstX), _mm512_set1_ps(rowYc));

                __m512 absW1 = _mm512_abs_ps(srcW1);
                __mmask16 singularMask = _mm512_cmp_ps_mask(absW1, epsilon, _CMP_LT_OQ);

                srcX = _mm512_div_ps(srcX, srcW1);
                srcY = _mm512_div_ps(srcY, srcW1);

                __m512 sx0f = _mm512_floor_ps(srcX);
                __m512 sy0f = _mm512_floor_ps(srcY);
                __m512 fxVec = _mm512_sub_ps(srcX, sx0f);
                __m512 fyVec = _mm512_sub_ps(srcY, sy0f);

                alignas(64) int sx0Arr[kLanes], sy0Arr[kLanes];
                alignas(64) float fxArr[kLanes], fyArr[kLanes];
                _mm512_store_si512(reinterpret_cast<void*>(sx0Arr), _mm512_cvttps_epi32(sx0f));
                _mm512_store_si512(reinterpret_cast<void*>(sy0Arr), _mm512_cvttps_epi32(sy0f));
                _mm512_store_ps(fxArr, fxVec);
                _mm512_store_ps(fyArr, fyVec);

                for (int lane = 0; lane < kLanes; ++lane)
                {
                    unsigned short* dstPixel = dstRow + static_cast<size_t>(x + lane) * 3;
                    bool singular = (singularMask & (1u << lane)) != 0;
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
