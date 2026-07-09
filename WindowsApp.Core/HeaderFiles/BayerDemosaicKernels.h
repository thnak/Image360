#pragma once

#include <algorithm>
#include <cstdint>

namespace WindowsApp::Core::Kernels
{
    namespace Detail
    {
        inline int BayerColor(int row, int col, uint32_t filters)
        {
            return (filters >> (((row << 1 & 14) | (col & 1)) << 1)) & 3;
        }

        // Bilinear Bayer demosaic for one pixel - matches
        // WindowsApp::Compute::Kernels::DemosaicBayerKernel's algorithm
        // exactly (this pixel's own CFA sample is exact; the other two
        // colors are averaged from same-color 3x3 neighbors, border pixels
        // naturally averaging over fewer neighbors). Irregular per-pixel
        // neighbor counts (border effects) make this a poor SIMD candidate
        // relative to its cost, so it's plain scalar in every tier -
        // BlackLevelSubtract/WhiteBalance (uniform elementwise arithmetic
        // over the full CFA plane) are where each tier's vectorization
        // actually lives.
        inline void DemosaicPixel(const unsigned short* cfaData, int width, int height,
                                   int x, int y, uint32_t filters, unsigned short* outPixel)
        {
            float rgb[3] = { 0.0f, 0.0f, 0.0f };
            int counts[3] = { 0, 0, 0 };

            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    int nx = x + dx;
                    int ny = y + dy;
                    if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;

                    int nChannel = BayerColor(ny, nx, filters);
                    int slot = (nChannel == 3) ? 1 : nChannel;
                    rgb[slot] += static_cast<float>(cfaData[static_cast<size_t>(ny) * width + nx]);
                    counts[slot]++;
                }
            }

            int myChannel = BayerColor(y, x, filters);
            int mySlot = (myChannel == 3) ? 1 : myChannel;

            for (int c = 0; c < 3; ++c)
            {
                if (c == mySlot || counts[c] == 0) continue;
                rgb[c] /= counts[c];
            }
            rgb[mySlot] = static_cast<float>(cfaData[static_cast<size_t>(y) * width + x]);

            for (int c = 0; c < 3; ++c)
            {
                outPixel[c] = static_cast<unsigned short>((std::min)((std::max)(rgb[c], 0.0f), 65535.0f));
            }
        }

        // Camera RGB -> sRGB 3x4 matrix apply for one pixel - matches
        // WindowsApp::Compute::Kernels::ColorMatrixKernel exactly.
        inline void ApplyColorMatrixPixel(const unsigned short* rgbIn, const float rgbCam[3][4], unsigned short* rgbOut)
        {
            float r = static_cast<float>(rgbIn[0]);
            float g = static_cast<float>(rgbIn[1]);
            float b = static_cast<float>(rgbIn[2]);

            for (int row = 0; row < 3; ++row)
            {
                float val = rgbCam[row][0] * r + rgbCam[row][1] * g + rgbCam[row][2] * b + rgbCam[row][3];
                rgbOut[row] = static_cast<unsigned short>((std::min)((std::max)(val, 0.0f), 65535.0f));
            }
        }
    }

    // Full RawIngest demosaic pipeline (BlackLevelSubtract -> WhiteBalance
    // -> bilinear DemosaicBayer -> camera-RGB->sRGB ColorMatrix; the GPU
    // kernel's final ToneCurve stage is an identity passthrough for v1, so
    // it's simply not applied rather than implemented as a no-op copy) -
    // matches CudaPipeline::DemosaicBayer's orchestration order exactly.
    // cfaData: one sample/pixel. rgbOut: pre-allocated width*height*3.
    namespace Scalar
    {
        void DemosaicBayer(const unsigned short* cfaData, int width, int height,
                            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
                            uint32_t filters, unsigned short* rgbOut);
    }
    namespace Avx2
    {
        void DemosaicBayer(const unsigned short* cfaData, int width, int height,
                            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
                            uint32_t filters, unsigned short* rgbOut);
    }
    namespace Avx512
    {
        void DemosaicBayer(const unsigned short* cfaData, int width, int height,
                            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
                            uint32_t filters, unsigned short* rgbOut);
    }
}
