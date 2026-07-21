#include "pch.h"
#include "HeaderFiles/ExposureFusionKernel.h"

#include <algorithm>
#include <cmath>

namespace WindowsApp::Core::Kernels::ExposureFusion
{
    namespace
    {
        // Separable 5-tap binomial blur ([1,4,6,4,1]/16), replicate
        // border - same math as SeamBlendKernels.h's Blur5Tap, reimplemented
        // locally (see this kernel's header comment on why).
        void Blur5Tap(const std::vector<float>& src, int w, int h, int channels, std::vector<float>& dst)
        {
            std::vector<float> tmp(src.size(), 0.0f);
            dst.assign(src.size(), 0.0f);
            const float k0 = 1.0f / 16.0f, k1 = 4.0f / 16.0f, k2 = 6.0f / 16.0f;

            auto clampX = [w](int x) { return (std::max)(0, (std::min)(w - 1, x)); };
            auto clampY = [h](int y) { return (std::max)(0, (std::min)(h - 1, y)); };

            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    size_t base = (static_cast<size_t>(y) * w + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        tmp[base + c] =
                            k0 * src[(static_cast<size_t>(y) * w + clampX(x - 2)) * channels + c] +
                            k1 * src[(static_cast<size_t>(y) * w + clampX(x - 1)) * channels + c] +
                            k2 * src[(static_cast<size_t>(y) * w + clampX(x)) * channels + c] +
                            k1 * src[(static_cast<size_t>(y) * w + clampX(x + 1)) * channels + c] +
                            k0 * src[(static_cast<size_t>(y) * w + clampX(x + 2)) * channels + c];
                    }
                }
            }
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    size_t base = (static_cast<size_t>(y) * w + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        dst[base + c] =
                            k0 * tmp[(static_cast<size_t>(clampY(y - 2)) * w + x) * channels + c] +
                            k1 * tmp[(static_cast<size_t>(clampY(y - 1)) * w + x) * channels + c] +
                            k2 * tmp[(static_cast<size_t>(clampY(y)) * w + x) * channels + c] +
                            k1 * tmp[(static_cast<size_t>(clampY(y + 1)) * w + x) * channels + c] +
                            k0 * tmp[(static_cast<size_t>(clampY(y + 2)) * w + x) * channels + c];
                    }
                }
            }
        }

        void Downsample2x(const std::vector<float>& src, int w, int h, int channels,
                           std::vector<float>& dst, int& outW, int& outH)
        {
            std::vector<float> blurred;
            Blur5Tap(src, w, h, channels, blurred);
            outW = (w + 1) / 2;
            outH = (h + 1) / 2;
            dst.assign(static_cast<size_t>(outW) * outH * channels, 0.0f);
            for (int y = 0; y < outH; ++y)
            {
                int sy = (std::min)(y * 2, h - 1);
                for (int x = 0; x < outW; ++x)
                {
                    int sx = (std::min)(x * 2, w - 1);
                    size_t dstBase = (static_cast<size_t>(y) * outW + x) * channels;
                    size_t srcBase = (static_cast<size_t>(sy) * w + sx) * channels;
                    for (int c = 0; c < channels; ++c) dst[dstBase + c] = blurred[srcBase + c];
                }
            }
        }

        void Upsample(const std::vector<float>& src, int w, int h, int channels,
                      int outW, int outH, std::vector<float>& dst)
        {
            dst.assign(static_cast<size_t>(outW) * outH * channels, 0.0f);
            float sx = (w > 1) ? static_cast<float>(w - 1) / static_cast<float>((std::max)(1, outW - 1)) : 0.0f;
            float sy = (h > 1) ? static_cast<float>(h - 1) / static_cast<float>((std::max)(1, outH - 1)) : 0.0f;
            for (int y = 0; y < outH; ++y)
            {
                float fy = sy * y;
                int y0 = (std::min)(static_cast<int>(fy), h - 1);
                int y1 = (std::min)(y0 + 1, h - 1);
                float wy = fy - y0;
                for (int x = 0; x < outW; ++x)
                {
                    float fx = sx * x;
                    int x0 = (std::min)(static_cast<int>(fx), w - 1);
                    int x1 = (std::min)(x0 + 1, w - 1);
                    float wx = fx - x0;
                    size_t dstBase = (static_cast<size_t>(y) * outW + x) * channels;
                    for (int c = 0; c < channels; ++c)
                    {
                        float v00 = src[(static_cast<size_t>(y0) * w + x0) * channels + c];
                        float v10 = src[(static_cast<size_t>(y0) * w + x1) * channels + c];
                        float v01 = src[(static_cast<size_t>(y1) * w + x0) * channels + c];
                        float v11 = src[(static_cast<size_t>(y1) * w + x1) * channels + c];
                        dst[dstBase + c] = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v10) +
                                            wy * ((1.0f - wx) * v01 + wx * v11);
                    }
                }
            }
        }

        struct PyramidLevel
        {
            std::vector<float> data;
            int w = 0, h = 0;
        };

        void BuildGaussianPyramid(const std::vector<float>& base, int w, int h, int channels,
                                   int numBands, std::vector<PyramidLevel>& outPyramid)
        {
            outPyramid.resize(numBands);
            outPyramid[0].data = base;
            outPyramid[0].w = w;
            outPyramid[0].h = h;
            for (int level = 1; level < numBands; ++level)
            {
                int outW = 0, outH = 0;
                Downsample2x(outPyramid[level - 1].data, outPyramid[level - 1].w, outPyramid[level - 1].h,
                             channels, outPyramid[level].data, outW, outH);
                outPyramid[level].w = outW;
                outPyramid[level].h = outH;
            }
        }

        void BuildLaplacianFromGaussian(const std::vector<PyramidLevel>& gaussian, int channels,
                                         std::vector<PyramidLevel>& outLaplacian)
        {
            int numBands = static_cast<int>(gaussian.size());
            outLaplacian.resize(numBands);
            for (int level = 0; level < numBands - 1; ++level)
            {
                std::vector<float> upsampled;
                Upsample(gaussian[level + 1].data, gaussian[level + 1].w, gaussian[level + 1].h, channels,
                         gaussian[level].w, gaussian[level].h, upsampled);
                size_t n = static_cast<size_t>(gaussian[level].w) * gaussian[level].h * channels;
                outLaplacian[level].data.assign(n, 0.0f);
                outLaplacian[level].w = gaussian[level].w;
                outLaplacian[level].h = gaussian[level].h;
                for (size_t p = 0; p < n; ++p) outLaplacian[level].data[p] = gaussian[level].data[p] - upsampled[p];
            }
            outLaplacian[numBands - 1] = gaussian[numBands - 1];
        }

        // exp(-((v-0.5)^2)/(2*sigma^2)), sigma=0.2 - the standard Mertens
        // well-exposedness constant (peaks at mid-gray, falls off toward
        // black/white clipping).
        inline float WellExposedness(float normalized)
        {
            constexpr float kSigma = 0.2f;
            float d = normalized - 0.5f;
            return std::exp(-(d * d) / (2.0f * kSigma * kSigma));
        }
    }

    void FuseTwoExposures(
        const unsigned short* low, const unsigned short* high,
        int width, int height, int numBands,
        std::vector<unsigned short>& outResult)
    {
        size_t count = static_cast<size_t>(width) * height;
        outResult.assign(count * 3, 0);
        if (width <= 0 || height <= 0) return;

        int maxBands = 1;
        { int w = width, h = height; while (w > 1 && h > 1 && maxBands < numBands) { w = (w + 1) / 2; h = (h + 1) / 2; ++maxBands; } }
        numBands = (std::min)(numBands, (std::max)(1, maxBands));

        std::vector<float> weightLow(count), weightHigh(count);
        constexpr float kEps = 1e-6f;
        for (size_t p = 0; p < count; ++p)
        {
            float wl = 1.0f, wh = 1.0f;
            for (int c = 0; c < 3; ++c)
            {
                wl *= WellExposedness(static_cast<float>(low[p * 3 + c]) / 65535.0f);
                wh *= WellExposedness(static_cast<float>(high[p * 3 + c]) / 65535.0f);
            }
            float sum = wl + wh;
            if (sum > kEps) { weightLow[p] = wl / sum; weightHigh[p] = wh / sum; }
            else { weightLow[p] = 0.5f; weightHigh[p] = 0.5f; }
        }

        std::vector<PyramidLevel> weightLowPyr, weightHighPyr;
        BuildGaussianPyramid(weightLow, width, height, 1, numBands, weightLowPyr);
        BuildGaussianPyramid(weightHigh, width, height, 1, numBands, weightHighPyr);

        std::vector<float> lowFloat(count * 3), highFloat(count * 3);
        for (size_t p = 0; p < count * 3; ++p)
        {
            lowFloat[p] = static_cast<float>(low[p]);
            highFloat[p] = static_cast<float>(high[p]);
        }

        std::vector<PyramidLevel> lowGaussian, highGaussian;
        BuildGaussianPyramid(lowFloat, width, height, 3, numBands, lowGaussian);
        BuildGaussianPyramid(highFloat, width, height, 3, numBands, highGaussian);

        std::vector<PyramidLevel> lowLaplacian, highLaplacian;
        BuildLaplacianFromGaussian(lowGaussian, 3, lowLaplacian);
        BuildLaplacianFromGaussian(highGaussian, 3, highLaplacian);

        std::vector<PyramidLevel> blended(numBands);
        for (int level = 0; level < numBands; ++level)
        {
            int lw = lowLaplacian[level].w, lh = lowLaplacian[level].h;
            blended[level].w = lw;
            blended[level].h = lh;
            size_t levelCount = static_cast<size_t>(lw) * lh;
            blended[level].data.assign(levelCount * 3, 0.0f);
            for (size_t p = 0; p < levelCount; ++p)
            {
                float wl = weightLowPyr[level].data[p];
                float wh = weightHighPyr[level].data[p];
                for (int c = 0; c < 3; ++c)
                    blended[level].data[p * 3 + c] =
                        wl * lowLaplacian[level].data[p * 3 + c] + wh * highLaplacian[level].data[p * 3 + c];
            }
        }

        std::vector<float> recon = blended[numBands - 1].data;
        int reconW = blended[numBands - 1].w, reconH = blended[numBands - 1].h;
        for (int level = numBands - 2; level >= 0; --level)
        {
            std::vector<float> upsampled;
            Upsample(recon, reconW, reconH, 3, blended[level].w, blended[level].h, upsampled);
            size_t n = static_cast<size_t>(blended[level].w) * blended[level].h * 3;
            recon.assign(n, 0.0f);
            for (size_t p = 0; p < n; ++p) recon[p] = blended[level].data[p] + upsampled[p];
            reconW = blended[level].w;
            reconH = blended[level].h;
        }

        for (size_t p = 0; p < count; ++p)
        {
            for (int c = 0; c < 3; ++c)
            {
                float v = recon[p * 3 + c];
                v = (std::max)(0.0f, (std::min)(65535.0f, v));
                outResult[p * 3 + c] = static_cast<unsigned short>(v + 0.5f);
            }
        }
    }
}
