#include "pch.h"
#include "HeaderFiles/PainterlyToneCurveKernel.h"

#include <algorithm>
#include <cmath>

namespace WindowsApp::Core::Kernels::PainterlyToneCurve
{
    void Apply(const unsigned short* src, int width, int height,
               float shadowGamma, float highlightRolloff, float vignetteStrength,
               std::vector<unsigned short>& outDst)
    {
        outDst.resize(static_cast<size_t>(width) * height * 3);

        float cx = (width - 1) * 0.5f;
        float cy = (height - 1) * 0.5f;
        float maxDist = std::sqrt(cx * cx + cy * cy);
        if (maxDist <= 0.0f) maxDist = 1.0f;

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                float dx = static_cast<float>(x) - cx;
                float dy = static_cast<float>(y) - cy;
                float r = std::sqrt(dx * dx + dy * dy) / maxDist;
                float vignette = 1.0f - vignetteStrength * r * r;

                size_t base = (static_cast<size_t>(y) * width + x) * 3;
                for (int c = 0; c < 3; ++c)
                {
                    float normalized = static_cast<float>(src[base + c]) / 65535.0f;
                    float shadowCrushed = std::pow((std::max)(0.0f, normalized), shadowGamma);
                    float rolledOff = shadowCrushed / (1.0f + highlightRolloff * shadowCrushed);
                    float mapped = rolledOff * vignette * 65535.0f;
                    outDst[base + c] = static_cast<unsigned short>(
                        (std::max)(0.0f, (std::min)(65535.0f, mapped)) + 0.5f);
                }
            }
        }
    }
}
