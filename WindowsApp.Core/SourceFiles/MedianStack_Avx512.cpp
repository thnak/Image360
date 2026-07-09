#include "pch.h"
#include "HeaderFiles/MedianStackKernels.h"

#include <algorithm>
#include <immintrin.h>

namespace WindowsApp::Core::Kernels::Avx512
{
    namespace
    {
        constexpr int kLanes = 32; // 32 x u16 = 512 bits, matches MAX_INPUTS
        constexpr int kMaxInputs = 32;
    }

    // Same approach as the AVX2 tier (see MedianStack_Avx2.cpp), widened to
    // 32 lanes per batch.
    void MedianStack(const unsigned short** inputs, int numInputs,
                      unsigned short* output, int width, int height, float sigmaThreshold)
    {
        int count = (std::min)(numInputs, kMaxInputs);
        long long numScalars = static_cast<long long>(width) * height * 3;

        alignas(64) unsigned short valuesBuf[kMaxInputs][kLanes];

        long long idx = 0;
        for (; idx + kLanes <= numScalars; idx += kLanes)
        {
            __m512 sumLo = _mm512_setzero_ps(), sumHi = _mm512_setzero_ps();
            __m512 sumSqLo = _mm512_setzero_ps(), sumSqHi = _mm512_setzero_ps();

            for (int i = 0; i < count; ++i)
            {
                __m512i u16 = _mm512_loadu_si512(reinterpret_cast<const void*>(inputs[i] + idx));
                _mm512_store_si512(reinterpret_cast<void*>(valuesBuf[i]), u16);

                __m256i lo256 = _mm512_castsi512_si256(u16);
                __m256i hi256 = _mm512_extracti64x4_epi64(u16, 1);
                __m512 fLo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(lo256));
                __m512 fHi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(hi256));

                sumLo = _mm512_add_ps(sumLo, fLo);
                sumHi = _mm512_add_ps(sumHi, fHi);
                sumSqLo = _mm512_add_ps(sumSqLo, _mm512_mul_ps(fLo, fLo));
                sumSqHi = _mm512_add_ps(sumSqHi, _mm512_mul_ps(fHi, fHi));
            }

            alignas(64) float sumArr[kLanes], sumSqArr[kLanes];
            _mm512_store_ps(sumArr, sumLo);
            _mm512_store_ps(sumArr + 16, sumHi);
            _mm512_store_ps(sumSqArr, sumSqLo);
            _mm512_store_ps(sumSqArr + 16, sumSqHi);

            for (int lane = 0; lane < kLanes; ++lane)
            {
                unsigned short laneValues[kMaxInputs];
                for (int i = 0; i < count; ++i) laneValues[i] = valuesBuf[i][lane];

                output[idx + lane] = Detail::MedianGivenStats(laneValues, count, sumArr[lane], sumSqArr[lane], sigmaThreshold);
            }
        }

        // Scalar tail for the remainder (< kLanes elements).
        for (; idx < numScalars; ++idx)
        {
            unsigned short values[kMaxInputs];
            for (int i = 0; i < count; ++i) values[i] = inputs[i][idx];
            output[idx] = Detail::MedianOfSigmaClipped(values, count, sigmaThreshold);
        }
    }
}
