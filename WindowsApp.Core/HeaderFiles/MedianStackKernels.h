#pragma once

#include <algorithm>
#include <cmath>

namespace WindowsApp::Core::Kernels
{
    namespace Detail
    {
        // Shared scalar finish step, given already-computed sum/sumSq (the
        // tiered kernels compute these via SIMD across a batch of lanes;
        // the scalar tier computes them directly per-lane): sigma-clip
        // `values[0..count)` in place (if count > 3) around the
        // mean/std those imply, then return the median of what remains.
        // Plain scalar code (no intrinsics) so it's safe to inline into
        // any tier's translation unit regardless of that TU's -march
        // flags. `count` is always <= 32 (MedianStack's own contributor
        // cap), so this is cheap however it's implemented.
        inline unsigned short MedianGivenStats(unsigned short* values, int count, float sum, float sumSq, float sigmaThreshold)
        {
            if (count > 3)
            {
                float mean = sum / count;
                float variance = sumSq / count - mean * mean;
                float sigma = std::sqrt((std::max)(variance, 0.0f));
                float lower = mean - sigmaThreshold * sigma;
                float upper = mean + sigmaThreshold * sigma;

                int validCount = 0;
                for (int i = 0; i < count; ++i)
                {
                    float v = static_cast<float>(values[i]);
                    if (v >= lower && v <= upper) values[validCount++] = values[i];
                }
                if (validCount > 0) count = validCount;
            }

            std::sort(values, values + count);
            return values[count / 2];
        }

        inline unsigned short MedianOfSigmaClipped(unsigned short* values, int count, float sigmaThreshold)
        {
            float sum = 0.0f, sumSq = 0.0f;
            if (count > 3)
            {
                for (int i = 0; i < count; ++i)
                {
                    float v = static_cast<float>(values[i]);
                    sum += v;
                    sumSq += v * v;
                }
            }
            return MedianGivenStats(values, count, sum, sumSq, sigmaThreshold);
        }
    }

    // Sigma-clipped median stack, matches
    // WindowsApp::Compute::Kernels::MedianStackKernel's algorithm exactly:
    // each of width*height*3 scalar values is stacked independently across
    // up to 32 inputs (numInputs capped to 32, extra inputs ignored -
    // matches CudaPipeline::MedianStack's own numInputs>32 rejection at
    // the caller level), sigma-clipped (if count>3) around the per-scalar
    // mean/std, then the median of what remains is taken. The exact
    // sorting algorithm doesn't matter (any correct sort yields the same
    // median), so this doesn't replicate the GPU's specific sorting
    // network bit-for-bit - only the sigma-clip statistics and the final
    // median value need to match.
    namespace Scalar
    {
        void MedianStack(const unsigned short** inputs, int numInputs,
                          unsigned short* output, int width, int height, float sigmaThreshold);
    }
    namespace Avx2
    {
        void MedianStack(const unsigned short** inputs, int numInputs,
                          unsigned short* output, int width, int height, float sigmaThreshold);
    }
    namespace Avx512
    {
        void MedianStack(const unsigned short** inputs, int numInputs,
                          unsigned short* output, int width, int height, float sigmaThreshold);
    }
}
