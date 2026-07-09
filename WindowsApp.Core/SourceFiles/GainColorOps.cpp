#include "pch.h"
#include "HeaderFiles/GainColorOps.h"

#include <algorithm>
#include <cmath>

namespace WindowsApp::Core
{
    // Plain scalar loop, not tiered - this file compiles without any
    // -march flag (unlike MedianStack/WarpPerspective/BayerDemosaic), so it
    // runs correctly on every CPU with no runtime dispatch needed. Gain is
    // a trivial multiply+clamp; the compiler's own auto-vectorizer already
    // handles it reasonably at -O2, and Lab conversion below is
    // transcendental-math-bound (pow/cbrt), not something a hand-rolled
    // AVX2 path would meaningfully speed up without a SIMD math library.
    void ApplyGainCpu(unsigned short* data, int numPixels, float gain)
    {
        if (!data || numPixels <= 0) return;
        if (std::fabs(gain - 1.0f) < 1e-6f) return; // no-op, matches CudaPipeline::ApplyGain

        for (int i = 0; i < numPixels; ++i)
        {
            float val = static_cast<float>(data[i]) * gain;
            data[i] = static_cast<unsigned short>((std::min)((std::max)(val, 0.0f), 65535.0f));
        }
    }

    namespace
    {
        float SrgbToLinear(float c)
        {
            return (c <= 0.04045f) ? (c / 12.92f) : std::pow((c + 0.055f) / 1.055f, 2.4f);
        }

        float LinearToSrgb(float c)
        {
            return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f);
        }

        float LabF(float t)
        {
            const float delta = 6.0f / 29.0f;
            return (t > delta * delta * delta) ? std::cbrt(t) : (t / (3.0f * delta * delta) + 4.0f / 29.0f);
        }

        float LabFInv(float t)
        {
            const float delta = 6.0f / 29.0f;
            return (t > delta) ? (t * t * t) : (3.0f * delta * delta * (t - 4.0f / 29.0f));
        }

        constexpr float kXn = 0.95047f;
        constexpr float kYn = 1.0f;
        constexpr float kZn = 1.08883f;

        void RgbPixelToLab(const unsigned short* rgb, float* lab)
        {
            float r = SrgbToLinear(rgb[0] / 65535.0f);
            float g = SrgbToLinear(rgb[1] / 65535.0f);
            float b = SrgbToLinear(rgb[2] / 65535.0f);

            float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
            float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
            float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

            float fx = LabF(X / kXn);
            float fy = LabF(Y / kYn);
            float fz = LabF(Z / kZn);

            lab[0] = 116.0f * fy - 16.0f;
            lab[1] = 500.0f * (fx - fy);
            lab[2] = 200.0f * (fy - fz);
        }

        void LabPixelToRgb(const float* lab, unsigned short* rgb)
        {
            float L = lab[0];
            float a = lab[1];
            float b = lab[2];

            float fy = (L + 16.0f) / 116.0f;
            float fx = fy + a / 500.0f;
            float fz = fy - b / 200.0f;

            float X = kXn * LabFInv(fx);
            float Y = kYn * LabFInv(fy);
            float Z = kZn * LabFInv(fz);

            float rl = 3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
            float gl = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
            float bl = 0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

            float r = LinearToSrgb((std::max)(rl, 0.0f));
            float g = LinearToSrgb((std::max)(gl, 0.0f));
            float bch = LinearToSrgb((std::max)(bl, 0.0f));

            rgb[0] = static_cast<unsigned short>((std::min)((std::max)(r * 65535.0f, 0.0f), 65535.0f));
            rgb[1] = static_cast<unsigned short>((std::min)((std::max)(g * 65535.0f, 0.0f), 65535.0f));
            rgb[2] = static_cast<unsigned short>((std::min)((std::max)(bch * 65535.0f, 0.0f), 65535.0f));
        }
    }

    void ComputeLabStatsCpu(const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3])
    {
        if (!rgb || !outMean || !outStd || width <= 0 || height <= 0) return;

        int numPixels = width * height;
        double sum[3] = { 0.0, 0.0, 0.0 };
        double sumSq[3] = { 0.0, 0.0, 0.0 };

        for (int i = 0; i < numPixels; ++i)
        {
            float lab[3];
            RgbPixelToLab(rgb + static_cast<size_t>(i) * 3, lab);
            for (int c = 0; c < 3; ++c)
            {
                double v = static_cast<double>(lab[c]);
                sum[c] += v;
                sumSq[c] += v * v;
            }
        }

        for (int c = 0; c < 3; ++c)
        {
            double mean = sum[c] / numPixels;
            double variance = sumSq[c] / numPixels - mean * mean;
            outMean[c] = mean;
            outStd[c] = std::sqrt((variance > 0.0) ? variance : 0.0);
        }
    }

    void ApplyReinhardColorTransferCpu(
        unsigned short* rgbInOut, int width, int height,
        const double srcMean[3], const double srcStd[3],
        const double refMean[3], const double refStd[3])
    {
        if (!rgbInOut || !srcMean || !srcStd || !refMean || !refStd || width <= 0 || height <= 0) return;

        int numPixels = width * height;
        for (int i = 0; i < numPixels; ++i)
        {
            unsigned short* px = rgbInOut + static_cast<size_t>(i) * 3;
            float lab[3];
            RgbPixelToLab(px, lab);

            for (int c = 0; c < 3; ++c)
            {
                double val = static_cast<double>(lab[c]);
                double stdSrc = (srcStd[c] > 1e-6) ? srcStd[c] : 1.0;
                double transferred = (val - srcMean[c]) * (refStd[c] / stdSrc) + refMean[c];
                lab[c] = static_cast<float>(transferred);
            }

            LabPixelToRgb(lab, px);
        }
    }
}
