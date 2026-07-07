#include "HeaderFiles/color_transfer.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace WindowsApp
{
    namespace Compute
    {
        namespace Kernels
        {
            __device__ __forceinline__ float SrgbToLinear(float c)
            {
                return (c <= 0.04045f) ? (c / 12.92f) : powf((c + 0.055f) / 1.055f, 2.4f);
            }

            __device__ __forceinline__ float LinearToSrgb(float c)
            {
                return (c <= 0.0031308f) ? (c * 12.92f) : (1.055f * powf(c, 1.0f / 2.4f) - 0.055f);
            }

            __device__ __forceinline__ float LabF(float t)
            {
                const float delta = 6.0f / 29.0f;
                return (t > delta * delta * delta) ? cbrtf(t) : (t / (3.0f * delta * delta) + 4.0f / 29.0f);
            }

            __device__ __forceinline__ float LabFInv(float t)
            {
                const float delta = 6.0f / 29.0f;
                return (t > delta) ? (t * t * t) : (3.0f * delta * delta * (t - 4.0f / 29.0f));
            }

            __global__ void RgbToLabKernel(
                const unsigned short* __restrict__ rgb, float* __restrict__ lab, int numPixels)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                int base = idx * 3;
                float r = SrgbToLinear(rgb[base] / 65535.0f);
                float g = SrgbToLinear(rgb[base + 1] / 65535.0f);
                float b = SrgbToLinear(rgb[base + 2] / 65535.0f);

                float X = 0.4124564f * r + 0.3575761f * g + 0.1804375f * b;
                float Y = 0.2126729f * r + 0.7151522f * g + 0.0721750f * b;
                float Z = 0.0193339f * r + 0.1191920f * g + 0.9503041f * b;

                const float Xn = 0.95047f;
                const float Yn = 1.0f;
                const float Zn = 1.08883f;

                float fx = LabF(X / Xn);
                float fy = LabF(Y / Yn);
                float fz = LabF(Z / Zn);

                lab[base]     = 116.0f * fy - 16.0f;
                lab[base + 1] = 500.0f * (fx - fy);
                lab[base + 2] = 200.0f * (fy - fz);
            }

            __global__ void LabToRgbKernel(
                const float* __restrict__ lab, unsigned short* __restrict__ rgb, int numPixels)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                int base = idx * 3;
                float L = lab[base];
                float a = lab[base + 1];
                float b = lab[base + 2];

                float fy = (L + 16.0f) / 116.0f;
                float fx = fy + a / 500.0f;
                float fz = fy - b / 200.0f;

                const float Xn = 0.95047f;
                const float Yn = 1.0f;
                const float Zn = 1.08883f;

                float X = Xn * LabFInv(fx);
                float Y = Yn * LabFInv(fy);
                float Z = Zn * LabFInv(fz);

                float rl =  3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z;
                float gl = -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z;
                float bl =  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z;

                float r = LinearToSrgb(fmaxf(rl, 0.0f));
                float g = LinearToSrgb(fmaxf(gl, 0.0f));
                float bch = LinearToSrgb(fmaxf(bl, 0.0f));

                rgb[base]     = static_cast<unsigned short>(fminf(fmaxf(r * 65535.0f, 0.0f), 65535.0f));
                rgb[base + 1] = static_cast<unsigned short>(fminf(fmaxf(g * 65535.0f, 0.0f), 65535.0f));
                rgb[base + 2] = static_cast<unsigned short>(fminf(fmaxf(bch * 65535.0f, 0.0f), 65535.0f));
            }

            __global__ void LabStatsKernel(
                const float* __restrict__ lab, int numPixels,
                double* __restrict__ outSum, double* __restrict__ outSumSq)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                for (int c = 0; c < 3; ++c)
                {
                    double v = static_cast<double>(lab[idx * 3 + c]);
                    atomicAdd(&outSum[c], v);
                    atomicAdd(&outSumSq[c], v * v);
                }
            }

            __global__ void ReinhardTransferKernel(
                float* __restrict__ lab, int numPixels,
                const double* __restrict__ srcMean, const double* __restrict__ srcStd,
                const double* __restrict__ refMean, const double* __restrict__ refStd)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                for (int c = 0; c < 3; ++c)
                {
                    double val = static_cast<double>(lab[idx * 3 + c]);
                    double stdSrc = (srcStd[c] > 1e-6) ? srcStd[c] : 1.0;
                    double transferred = (val - srcMean[c]) * (refStd[c] / stdSrc) + refMean[c];
                    lab[idx * 3 + c] = static_cast<float>(transferred);
                }
            }
        }
    }
}
