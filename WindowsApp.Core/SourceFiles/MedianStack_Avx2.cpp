#include "pch.h"
#include "HeaderFiles/MedianStackKernels.h"

#include <algorithm>
#include <immintrin.h>

namespace WindowsApp::Core::Kernels::Avx2
{
    namespace
    {
        constexpr int kLanes = 16; // 16 x u16 = 256 bits
        constexpr int kMaxInputs = 32;
    }

    // Vectorizes the O(numInputs) sum/sumSq reduction across a batch of 16
    // output scalars at once (the dominant cost when numInputs is large -
    // e.g. 32 full-resolution contributor images) - the compaction+sort
    // finish is inherently per-lane/data-dependent, so it stays scalar via
    // the shared Detail::MedianGivenStats helper, fed the SIMD-computed
    // stats instead of recomputing them.
    void MedianStack(const unsigned short** inputs, int numInputs,
                      unsigned short* output, int width, int height, float sigmaThreshold)
    {
        int count = (std::min)(numInputs, kMaxInputs);
        long long numScalars = static_cast<long long>(width) * height * 3;

        alignas(32) unsigned short valuesBuf[kMaxInputs][kLanes];

        long long idx = 0;
        for (; idx + kLanes <= numScalars; idx += kLanes)
        {
            __m256 sumLo = _mm256_setzero_ps(), sumHi = _mm256_setzero_ps();
            __m256 sumSqLo = _mm256_setzero_ps(), sumSqHi = _mm256_setzero_ps();

            for (int i = 0; i < count; ++i)
            {
                __m256i u16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(inputs[i] + idx));
                _mm256_store_si256(reinterpret_cast<__m256i*>(valuesBuf[i]), u16);

                __m128i lo128 = _mm256_castsi256_si128(u16);
                __m128i hi128 = _mm256_extracti128_si256(u16, 1);
                __m256 fLo = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(lo128));
                __m256 fHi = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(hi128));

                sumLo = _mm256_add_ps(sumLo, fLo);
                sumHi = _mm256_add_ps(sumHi, fHi);
                sumSqLo = _mm256_add_ps(sumSqLo, _mm256_mul_ps(fLo, fLo));
                sumSqHi = _mm256_add_ps(sumSqHi, _mm256_mul_ps(fHi, fHi));
            }

            alignas(32) float sumArr[kLanes], sumSqArr[kLanes];
            _mm256_store_ps(sumArr, sumLo);
            _mm256_store_ps(sumArr + 8, sumHi);
            _mm256_store_ps(sumSqArr, sumSqLo);
            _mm256_store_ps(sumSqArr + 8, sumSqHi);

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
