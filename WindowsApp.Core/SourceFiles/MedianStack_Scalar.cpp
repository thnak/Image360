#include "pch.h"
#include "HeaderFiles/MedianStackKernels.h"

#include <algorithm>

namespace WindowsApp::Core::Kernels::Scalar
{
    void MedianStack(const unsigned short** inputs, int numInputs,
                      unsigned short* output, int width, int height, float sigmaThreshold)
    {
        constexpr int kMaxInputs = 32;
        int count = (std::min)(numInputs, kMaxInputs);
        long long numScalars = static_cast<long long>(width) * height * 3;

        unsigned short values[kMaxInputs];
        for (long long idx = 0; idx < numScalars; ++idx)
        {
            for (int i = 0; i < count; ++i)
            {
                values[i] = inputs[i][idx];
            }
            output[idx] = Detail::MedianOfSigmaClipped(values, count, sigmaThreshold);
        }
    }
}
