#include "pch.h"
#include "HeaderFiles/BayerDemosaicKernels.h"

#include <algorithm>
#include <immintrin.h>
#include <vector>

namespace WindowsApp::Core::Kernels::Avx512
{
    namespace
    {
        constexpr int kLanes = 32;

        void BlackLevelSubtractVec(unsigned short* cfa, int numPixels, unsigned short blackLevel)
        {
            const __m512i blackVec = _mm512_set1_epi16(static_cast<short>(blackLevel));
            int i = 0;
            for (; i + kLanes <= numPixels; i += kLanes)
            {
                __m512i v = _mm512_loadu_si512(reinterpret_cast<const void*>(cfa + i));
                __m512i result = _mm512_subs_epu16(v, blackVec);
                _mm512_storeu_si512(reinterpret_cast<void*>(cfa + i), result);
            }
            for (; i < numPixels; ++i)
            {
                cfa[i] = (cfa[i] > blackLevel) ? (cfa[i] - blackLevel) : 0;
            }
        }

        // See BayerDemosaic_Avx2.cpp's WhiteBalanceRowVec - same 2-element
        // repeating gain pattern, widened to 16-wide float halves (32 x u16
        // per full vector).
        void WhiteBalanceRowVec(unsigned short* cfaRow, int width, float gainEven, float gainOdd)
        {
            const __m512 gainVec = _mm512_set_ps(
                gainOdd, gainEven, gainOdd, gainEven, gainOdd, gainEven, gainOdd, gainEven,
                gainOdd, gainEven, gainOdd, gainEven, gainOdd, gainEven, gainOdd, gainEven);
            const __m512 zero = _mm512_setzero_ps();
            const __m512 maxVal = _mm512_set1_ps(65535.0f);

            int x = 0;
            for (; x + kLanes <= width; x += kLanes)
            {
                __m512i u16 = _mm512_loadu_si512(reinterpret_cast<const void*>(cfaRow + x));
                __m256i lo256 = _mm512_castsi512_si256(u16);
                __m256i hi256 = _mm512_extracti64x4_epi64(u16, 1);
                __m512 fLo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(lo256));
                __m512 fHi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(hi256));

                fLo = _mm512_mul_ps(fLo, gainVec);
                fHi = _mm512_mul_ps(fHi, gainVec);
                fLo = _mm512_min_ps(_mm512_max_ps(fLo, zero), maxVal);
                fHi = _mm512_min_ps(_mm512_max_ps(fHi, zero), maxVal);

                __m256i iLo = _mm512_cvtusepi32_epi16(_mm512_cvttps_epu32(fLo));
                __m256i iHi = _mm512_cvtusepi32_epi16(_mm512_cvttps_epu32(fHi));

                _mm256_storeu_si256(reinterpret_cast<__m256i*>(cfaRow + x), iLo);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(cfaRow + x + 16), iHi);
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
