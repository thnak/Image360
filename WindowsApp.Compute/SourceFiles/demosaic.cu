#include "HeaderFiles/demosaic.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace WindowsApp
{
    namespace Compute
    {
        namespace Kernels
        {
            // Matches LibRaw's own COLOR(row,col) macro convention:
            // 0=R, 1=G, 2=B, 3=G2 (second green in RGGB-style patterns).
            __device__ __forceinline__ int BayerColor(int row, int col, uint32_t filters)
            {
                return (filters >> (((row << 1 & 14) | (col & 1)) << 1)) & 3;
            }

            // =====================================================================
            // Demosaic Kernel 1: Black Level Subtract
            // =====================================================================
            __global__ void BlackLevelSubtractKernel(
                unsigned short* __restrict__ cfaData,
                int numPixels, unsigned short blackLevel)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                unsigned short v = cfaData[idx];
                cfaData[idx] = (v > blackLevel) ? (v - blackLevel) : 0;
            }

            // =====================================================================
            // Demosaic Kernel 2: White Balance
            // =====================================================================
            __global__ void WhiteBalanceKernel(
                unsigned short* __restrict__ cfaData,
                int width, int height,
                float camMul0, float camMul1, float camMul2, float camMul3,
                uint32_t filters)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= width || y >= height)
                    return;

                int idx = y * width + x;
                int channel = BayerColor(y, x, filters);

                float gain;
                switch (channel)
                {
                case 0: gain = camMul0; break;
                case 1: gain = camMul1; break;
                case 2: gain = camMul2; break;
                default: gain = camMul3; break;
                }

                float val = static_cast<float>(cfaData[idx]) * gain;
                cfaData[idx] = static_cast<unsigned short>(fminf(fmaxf(val, 0.0f), 65535.0f));
            }

            // =====================================================================
            // Demosaic Kernel 3: Bayer Demosaic (bilinear)
            // =====================================================================
            // For each output pixel: use the exact sample for the color this
            // pixel's own CFA cell holds; bilinear-average same-color 3x3
            // neighbors for the other two colors. Channel 3 (second green,
            // per LibRaw's convention) folds into the green output slot.
            __global__ void DemosaicBayerKernel(
                const unsigned short* __restrict__ cfaData,
                unsigned short* __restrict__ rgbOut,
                int width, int height, uint32_t filters)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= width || y >= height)
                    return;

                float rgb[3] = { 0.0f, 0.0f, 0.0f };
                int counts[3] = { 0, 0, 0 };

                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dx = -1; dx <= 1; ++dx)
                    {
                        int nx = x + dx;
                        int ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                            continue;

                        int nChannel = BayerColor(ny, nx, filters);
                        int slot = (nChannel == 3) ? 1 : nChannel;
                        rgb[slot] += static_cast<float>(cfaData[ny * width + nx]);
                        counts[slot]++;
                    }
                }

                int myChannel = BayerColor(y, x, filters);
                int mySlot = (myChannel == 3) ? 1 : myChannel;

                for (int c = 0; c < 3; ++c)
                {
                    if (c == mySlot || counts[c] == 0)
                        continue;
                    rgb[c] /= counts[c];
                }
                // This pixel's own CFA sample is exact, not averaged.
                rgb[mySlot] = static_cast<float>(cfaData[y * width + x]);

                int outIdx = (y * width + x) * 3;
                for (int c = 0; c < 3; ++c)
                {
                    rgbOut[outIdx + c] = static_cast<unsigned short>(fminf(fmaxf(rgb[c], 0.0f), 65535.0f));
                }
            }

            // =====================================================================
            // Demosaic Kernel 4: Color Matrix (camera RGB -> sRGB)
            // =====================================================================
            __global__ void ColorMatrixKernel(
                const unsigned short* __restrict__ rgbIn,
                unsigned short* __restrict__ rgbOut,
                int numPixels,
                const float* __restrict__ rgbCam)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                int base = idx * 3;
                float r = static_cast<float>(rgbIn[base]);
                float g = static_cast<float>(rgbIn[base + 1]);
                float b = static_cast<float>(rgbIn[base + 2]);

                for (int row = 0; row < 3; ++row)
                {
                    float val = rgbCam[row * 4 + 0] * r + rgbCam[row * 4 + 1] * g
                              + rgbCam[row * 4 + 2] * b + rgbCam[row * 4 + 3];
                    rgbOut[base + row] = static_cast<unsigned short>(fminf(fmaxf(val, 0.0f), 65535.0f));
                }
            }

            // =====================================================================
            // Demosaic Kernel 5: Tone Curve (linear passthrough for v1)
            // =====================================================================
            __global__ void ToneCurveKernel(
                const unsigned short* __restrict__ rgbIn,
                unsigned short* __restrict__ rgbOut,
                int numPixels)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                int base = idx * 3;
                rgbOut[base] = rgbIn[base];
                rgbOut[base + 1] = rgbIn[base + 1];
                rgbOut[base + 2] = rgbIn[base + 2];
            }
        }
    }
}
