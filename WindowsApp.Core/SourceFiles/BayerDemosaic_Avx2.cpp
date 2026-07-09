#include "pch.h"
#include "HeaderFiles/BayerDemosaicKernels.h"

#include <algorithm>
#include <immintrin.h>
#include <vector>

namespace WindowsApp::Core::Kernels::Avx2
{
    namespace
    {
        constexpr int kLanes = 16;

        void BlackLevelSubtractVec(unsigned short* cfa, int numPixels, unsigned short blackLevel)
        {
            const __m256i blackVec = _mm256_set1_epi16(static_cast<short>(blackLevel));
            int i = 0;
            for (; i + kLanes <= numPixels; i += kLanes)
            {
                __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cfa + i));
                // Unsigned saturating subtract clamps to 0 for v<=blackLevel,
                // matching the scalar "(v>blackLevel) ? v-blackLevel : 0" exactly.
                __m256i result = _mm256_subs_epu16(v, blackVec);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(cfa + i), result);
            }
            for (; i < numPixels; ++i)
            {
                cfa[i] = (cfa[i] > blackLevel) ? (cfa[i] - blackLevel) : 0;
            }
        }

        // For a fixed row, Detail::BayerColor alternates between exactly 2
        // channel values as x parity flips (standard Bayer: each row is a
        // 2-color repeating pattern) - so the per-pixel gain for a whole
        // row is a 2-element repeating vector, broadcastable directly
        // instead of computing BayerColor per pixel.
        void WhiteBalanceRowVec(unsigned short* cfaRow, int width, float gainEven, float gainOdd)
        {
            const __m256 gainVec = _mm256_set_ps(gainOdd, gainEven, gainOdd, gainEven, gainOdd, gainEven, gainOdd, gainEven);
            const __m256 zero = _mm256_setzero_ps();
            const __m256 maxVal = _mm256_set1_ps(65535.0f);

            int x = 0;
            for (; x + kLanes <= width; x += kLanes)
            {
                __m256i u16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cfaRow + x));
                __m128i lo128 = _mm256_castsi256_si128(u16);
                __m128i hi128 = _mm256_extracti128_si256(u16, 1);
                __m256 fLo = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo128));
                __m256 fHi = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi128));

                fLo = _mm256_mul_ps(fLo, gainVec);
                fHi = _mm256_mul_ps(fHi, gainVec);
                fLo = _mm256_min_ps(_mm256_max_ps(fLo, zero), maxVal);
                fHi = _mm256_min_ps(_mm256_max_ps(fHi, zero), maxVal);

                __m256i iLo = _mm256_cvttps_epi32(fLo);
                __m256i iHi = _mm256_cvttps_epi32(fHi);
                __m128i packedLo = _mm_packus_epi32(_mm256_castsi256_si128(iLo), _mm256_extracti128_si256(iLo, 1));
                __m128i packedHi = _mm_packus_epi32(_mm256_castsi256_si128(iHi), _mm256_extracti128_si256(iHi, 1));

                _mm_storeu_si128(reinterpret_cast<__m128i*>(cfaRow + x), packedLo);
                _mm_storeu_si128(reinterpret_cast<__m128i*>(cfaRow + x + 8), packedHi);
            }
            for (; x < width; ++x)
            {
                float gain = (x % 2 == 0) ? gainEven : gainOdd;
                float val = static_cast<float>(cfaRow[x]) * gain;
                cfaRow[x] = static_cast<unsigned short>((std::min)((std::max)(val, 0.0f), 65535.0f));
            }
        }
    }

    void DemosaicBayer(const unsigned short* cfaData, int width, int height,
                        unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
                        uint32_t filters, unsigned short* rgbOut)
    {
        int numPixels = width * height;
        std::vector<unsigned short> cfa(cfaData, cfaData + numPixels);

        BlackLevelSubtractVec(cfa.data(), numPixels, blackLevel);

        for (int y = 0; y < height; ++y)
        {
            int evenChannel = Detail::BayerColor(y, 0, filters);
            int oddChannel = Detail::BayerColor(y, 1, filters);
            float gainEven = camMul[evenChannel];
            float gainOdd = camMul[oddChannel];
            WhiteBalanceRowVec(cfa.data() + static_cast<size_t>(y) * width, width, gainEven, gainOdd);
        }

        std::vector<unsigned short> demosaiced(static_cast<size_t>(numPixels) * 3);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Detail::DemosaicPixel(cfa.data(), width, height, x, y, filters,
                                       demosaiced.data() + (static_cast<size_t>(y) * width + x) * 3);
            }
        }

        for (int i = 0; i < numPixels; ++i)
        {
            Detail::ApplyColorMatrixPixel(demosaiced.data() + static_cast<size_t>(i) * 3, rgbCam, rgbOut + static_cast<size_t>(i) * 3);
        }
    }
}
