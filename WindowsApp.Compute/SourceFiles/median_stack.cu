#include "HeaderFiles/median_stack.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

namespace WindowsApp
{
    namespace Compute
    {
        namespace Kernels
        {
            // Block size for 2D kernels
            constexpr int BLOCK_W = 16;
            constexpr int BLOCK_H = 16;

            // Max inputs for median stack (limited by shared memory)
            constexpr int MAX_INPUTS = 32;

            // =====================================================================
            // Kernel 1: Perspective Warp (backward mapping with inverse homography)
            // =====================================================================
            // Each thread computes one output pixel by mapping back to source space.
            // Uses pitch-linear memory for coalesced access.
            __global__ void WarpPerspectiveKernel(
                const unsigned short *__restrict__ srcData,
                unsigned short *__restrict__ dstData,
                int srcW, int srcH, int dstW, int dstH,
                const float *__restrict__ invH,
                int offsetX, int offsetY)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= dstW || y >= dstH)
                    return;

                // Apply inverse homography: src = invH * dst
                float dstX = static_cast<float>(x + offsetX);
                float dstY = static_cast<float>(y + offsetY);

                float srcX = invH[0] * dstX + invH[1] * dstY + invH[2];
                float srcY = invH[3] * dstX + invH[4] * dstY + invH[5];
                float srcW1 = invH[6] * dstX + invH[7] * dstY + invH[8];

                int dstIdx = (y * dstW + x) * 3;

                if (fabsf(srcW1) < 1e-10f)
                {
                    dstData[dstIdx] = 0;
                    dstData[dstIdx + 1] = 0;
                    dstData[dstIdx + 2] = 0;
                    return;
                }

                srcX /= srcW1;
                srcY /= srcW1;

                // Bilinear interpolation
                int sx0 = __float2int_rd(srcX);
                int sy0 = __float2int_rd(srcY);
                int sx1 = sx0 + 1;
                int sy1 = sy0 + 1;

                float fx = srcX - sx0;
                float fy = srcY - sy0;

                // Bounds check
                if (sx0 < 0 || sy0 < 0 || sx1 >= srcW || sy1 >= srcH)
                {
                    dstData[dstIdx] = 0;
                    dstData[dstIdx + 1] = 0;
                    dstData[dstIdx + 2] = 0;
                    return;
                }

                // Bilinear weights
                float w00 = (1.0f - fx) * (1.0f - fy);
                float w10 = fx * (1.0f - fy);
                float w01 = (1.0f - fx) * fy;
                float w11 = fx * fy;

                // Sample 4 neighbors for each channel
                for (int c = 0; c < 3; c++)
                {
                    float v00 = static_cast<float>(srcData[(sy0 * srcW + sx0) * 3 + c]);
                    float v10 = static_cast<float>(srcData[(sy0 * srcW + sx1) * 3 + c]);
                    float v01 = static_cast<float>(srcData[(sy1 * srcW + sx0) * 3 + c]);
                    float v11 = static_cast<float>(srcData[(sy1 * srcW + sx1) * 3 + c]);

                    float val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                    dstData[dstIdx + c] = static_cast<unsigned short>(fminf(fmaxf(val, 0.0f), 65535.0f));
                }
            }

            // =====================================================================
            // Kernel 2: Median Stack with Sigma-Clipping
            // =====================================================================
            // Each thread handles one pixel channel across all input images.
            // Uses Batcher's Odd-Even Mergesort for register-friendly sorting.
            // Sorting network for N=8 (unrolled, branchless).
            __device__ void SortNetwork8(unsigned short *arr)
            {
// Batcher's Odd-Even Mergesort for 8 elements
// Fully unrolled, no branches, register-friendly
#define SWAP(a, b)                 \
    if (arr[a] > arr[b])           \
    {                              \
        unsigned short t = arr[a]; \
        arr[a] = arr[b];           \
        arr[b] = t;                \
    }
                SWAP(0, 1);
                SWAP(2, 3);
                SWAP(4, 5);
                SWAP(6, 7);
                SWAP(0, 2);
                SWAP(1, 3);
                SWAP(4, 6);
                SWAP(5, 7);
                SWAP(1, 2);
                SWAP(5, 6);
                SWAP(0, 4);
                SWAP(1, 5);
                SWAP(2, 6);
                SWAP(3, 7);
                SWAP(2, 4);
                SWAP(3, 5);
                SWAP(1, 2);
                SWAP(3, 4);
                SWAP(5, 6);
#undef SWAP
            }

            __device__ void SortNetwork16(unsigned short *arr)
            {
// Batcher's Odd-Even Mergesort for 16 elements
#define SWAP(a, b)                 \
    if (arr[a] > arr[b])           \
    {                              \
        unsigned short t = arr[a]; \
        arr[a] = arr[b];           \
        arr[b] = t;                \
    }
                // Odd-Even merge for 16
                SWAP(0, 1);
                SWAP(2, 3);
                SWAP(4, 5);
                SWAP(6, 7);
                SWAP(8, 9);
                SWAP(10, 11);
                SWAP(12, 13);
                SWAP(14, 15);
                SWAP(0, 2);
                SWAP(1, 3);
                SWAP(4, 6);
                SWAP(5, 7);
                SWAP(8, 10);
                SWAP(9, 11);
                SWAP(12, 14);
                SWAP(13, 15);
                SWAP(1, 2);
                SWAP(5, 6);
                SWAP(9, 10);
                SWAP(13, 14);
                SWAP(0, 4);
                SWAP(1, 5);
                SWAP(2, 6);
                SWAP(3, 7);
                SWAP(8, 12);
                SWAP(9, 13);
                SWAP(10, 14);
                SWAP(11, 15);
                SWAP(2, 4);
                SWAP(3, 5);
                SWAP(10, 12);
                SWAP(11, 13);
                SWAP(1, 2);
                SWAP(3, 4);
                SWAP(5, 6);
                SWAP(9, 10);
                SWAP(11, 12);
                SWAP(13, 14);
                SWAP(0, 8);
                SWAP(1, 9);
                SWAP(2, 10);
                SWAP(3, 11);
                SWAP(4, 12);
                SWAP(5, 13);
                SWAP(6, 14);
                SWAP(7, 15);
                SWAP(4, 8);
                SWAP(5, 9);
                SWAP(6, 10);
                SWAP(7, 11);
                SWAP(2, 4);
                SWAP(3, 5);
                SWAP(6, 8);
                SWAP(7, 9);
                SWAP(10, 12);
                SWAP(11, 13);
                SWAP(1, 2);
                SWAP(3, 4);
                SWAP(5, 6);
                SWAP(7, 8);
                SWAP(9, 10);
                SWAP(11, 12);
                SWAP(13, 14);
#undef SWAP
            }

            __global__ void MedianStackKernel(
                const unsigned short *__restrict__ inputs,
                unsigned short *__restrict__ output,
                int numInputs, int numPixels,
                float sigmaThreshold)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPixels)
                    return;

                // Collect values from all inputs for this pixel
                unsigned short values[MAX_INPUTS];
                int count = min(numInputs, MAX_INPUTS);

                for (int i = 0; i < count; i++)
                {
                    values[i] = inputs[i * numPixels + idx];
                }

                // Sigma-clipping: compute mean and std dev
                if (count > 3)
                {
                    float sum = 0.0f;
                    float sumSq = 0.0f;
                    for (int i = 0; i < count; i++)
                    {
                        float v = static_cast<float>(values[i]);
                        sum += v;
                        sumSq += v * v;
                    }

                    float mean = sum / count;
                    float variance = sumSq / count - mean * mean;
                    float sigma = sqrtf(fmaxf(variance, 0.0f));
                    float lower = mean - sigmaThreshold * sigma;
                    float upper = mean + sigmaThreshold * sigma;

                    // Compact: remove outliers
                    int validCount = 0;
                    for (int i = 0; i < count; i++)
                    {
                        float v = static_cast<float>(values[i]);
                        if (v >= lower && v <= upper)
                        {
                            values[validCount++] = values[i];
                        }
                    }

                    if (validCount > 0)
                        count = validCount;
                }

                // Sort using appropriate network
                if (count <= 8)
                {
                    // Pad to 8 for sorting network
                    for (int i = count; i < 8; i++)
                        values[i] = values[count - 1];
                    SortNetwork8(values);
                }
                else if (count <= 16)
                {
                    for (int i = count; i < 16; i++)
                        values[i] = values[count - 1];
                    SortNetwork16(values);
                }
                else
                {
                    // Insertion sort for larger counts (rare case)
                    for (int i = 1; i < count; i++)
                    {
                        unsigned short key = values[i];
                        int j = i - 1;
                        while (j >= 0 && values[j] > key)
                        {
                            values[j + 1] = values[j];
                            j--;
                        }
                        values[j + 1] = key;
                    }
                }

                // Output median
                output[idx] = values[count / 2];
            }

            // =====================================================================
            // Kernel 3: Gaussian Blur (separable, horizontal)
            // =====================================================================
            // Uses shared memory for coalesced access.
            // Each thread block loads a tile + halo into shared memory.
            __global__ void GaussianBlurHorizontal(
                const unsigned short *__restrict__ input,
                unsigned short *__restrict__ output,
                int width, int height,
                const float *__restrict__ kernel, int kernelRadius)
            {
                extern __shared__ unsigned short sdata[];

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                int sharedWidth = blockDim.x + 2 * kernelRadius;
                int localX = threadIdx.x + kernelRadius;

                // Load center pixel into shared memory
                if (x < width && y < height)
                {
                    sdata[threadIdx.y * sharedWidth + localX] = input[y * width + x];
                }

                // Load left halo
                if (threadIdx.x < kernelRadius)
                {
                    int haloX = blockIdx.x * blockDim.x + threadIdx.x - kernelRadius;
                    if (haloX >= 0 && y < height)
                        sdata[threadIdx.y * sharedWidth + threadIdx.x] = input[y * width + haloX];
                    else
                        sdata[threadIdx.y * sharedWidth + threadIdx.x] = 0;
                }

                // Load right halo
                if (threadIdx.x < kernelRadius)
                {
                    int haloX = blockIdx.x * blockDim.x + blockDim.x + threadIdx.x;
                    if (haloX < width && y < height)
                        sdata[threadIdx.y * sharedWidth + localX + blockDim.x] = input[y * width + haloX];
                    else
                        sdata[threadIdx.y * sharedWidth + localX + blockDim.x] = 0;
                }

                __syncthreads();

                if (x >= width || y >= height)
                    return;

                // Apply kernel
                float sum = 0.0f;
                for (int k = -kernelRadius; k <= kernelRadius; k++)
                {
                    sum += static_cast<float>(sdata[threadIdx.y * sharedWidth + localX + k]) * kernel[k + kernelRadius];
                }

                output[y * width + x] = static_cast<unsigned short>(fminf(fmaxf(sum, 0.0f), 65535.0f));
            }

            // =====================================================================
            // Kernel 3: Gaussian Blur (separable, vertical)
            // =====================================================================
            __global__ void GaussianBlurVertical(
                const unsigned short *__restrict__ input,
                unsigned short *__restrict__ output,
                int width, int height,
                const float *__restrict__ kernel, int kernelRadius)
            {
                extern __shared__ unsigned short sdata[];

                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                int sharedHeight = blockDim.y + 2 * kernelRadius;
                int localY = threadIdx.y + kernelRadius;

                // Load center pixel
                if (x < width && y < height)
                {
                    sdata[localY * blockDim.x + threadIdx.x] = input[y * width + x];
                }

                // Load top halo
                if (threadIdx.y < kernelRadius)
                {
                    int haloY = blockIdx.y * blockDim.y + threadIdx.y - kernelRadius;
                    if (haloY >= 0 && x < width)
                        sdata[threadIdx.y * blockDim.x + threadIdx.x] = input[haloY * width + x];
                    else
                        sdata[threadIdx.y * blockDim.x + threadIdx.x] = 0;
                }

                // Load bottom halo
                if (threadIdx.y < kernelRadius)
                {
                    int haloY = blockIdx.y * blockDim.y + blockDim.y + threadIdx.y;
                    if (haloY < height && x < width)
                        sdata[(localY + blockDim.y) * blockDim.x + threadIdx.x] = input[haloY * width + x];
                    else
                        sdata[(localY + blockDim.y) * blockDim.x + threadIdx.x] = 0;
                }

                __syncthreads();

                if (x >= width || y >= height)
                    return;

                // Apply kernel
                float sum = 0.0f;
                for (int k = -kernelRadius; k <= kernelRadius; k++)
                {
                    sum += static_cast<float>(sdata[(localY + k) * blockDim.x + threadIdx.x]) * kernel[k + kernelRadius];
                }

                output[y * width + x] = static_cast<unsigned short>(fminf(fmaxf(sum, 0.0f), 65535.0f));
            }

            // =====================================================================
            // Kernel 3: Laplacian Subtract (current - upsampled)
            // =====================================================================
            __global__ void LaplacianSubtract(
                const unsigned short *__restrict__ currentLevel,
                const unsigned short *__restrict__ upsampledLower,
                unsigned short *__restrict__ laplacianOut,
                int width, int height)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= width || y >= height)
                    return;

                int idx = y * width + x;
                int cur = static_cast<int>(currentLevel[idx]);
                int ups = static_cast<int>(upsampledLower[idx]);

                // Laplacian = current - upsampled (signed, biased to unsigned range)
                int diff = cur - ups + 32768;
                laplacianOut[idx] = static_cast<unsigned short>(min(max(diff, 0), 65535));
            }

            // =====================================================================
            // Kernel 3: Blend Laplacian Levels
            // =====================================================================
            __global__ void BlendLaplacianLevels(
                const unsigned short *__restrict__ lapA,
                const unsigned short *__restrict__ lapB,
                unsigned short *__restrict__ blended,
                int width, int height, float weightA, float weightB)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= width || y >= height)
                    return;

                int idx = y * width + x;
                float a = static_cast<float>(lapA[idx]) - 32768.0f; // Unbias
                float b = static_cast<float>(lapB[idx]) - 32768.0f;

                float val = a * weightA + b * weightB + 32768.0f;
                blended[idx] = static_cast<unsigned short>(fminf(fmaxf(val, 0.0f), 65535.0f));
            }

            // =====================================================================
            // Kernel 3: Laplacian Reconstruct
            // =====================================================================
            __global__ void LaplacianReconstruct(
                const unsigned short *__restrict__ laplacian,
                const unsigned short *__restrict__ upsampledLower,
                unsigned short *__restrict__ output,
                int width, int height)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= width || y >= height)
                    return;

                int idx = y * width + x;
                int lap = static_cast<int>(laplacian[idx]) - 32768; // Unbias
                int ups = static_cast<int>(upsampledLower[idx]);

                int reconstructed = lap + ups;
                output[idx] = static_cast<unsigned short>(min(max(reconstructed, 0), 65535));
            }

            // =====================================================================
            // Kernel 4: Gain Compensation
            // =====================================================================
            __global__ void ApplyGainKernel(
                unsigned short *__restrict__ data,
                int numValues, float gain)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numValues)
                    return;

                float val = static_cast<float>(data[idx]) * gain;
                data[idx] = static_cast<unsigned short>(fminf(fmaxf(val, 0.0f), 65535.0f));
            }

            // =====================================================================
            // Utility: Downsample 2x (box filter)
            // =====================================================================
            __global__ void Downsample2x(
                const unsigned short *__restrict__ input,
                unsigned short *__restrict__ output,
                int srcW, int srcH, int dstW, int dstH)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= dstW || y >= dstH)
                    return;

                int sx = x * 2;
                int sy = y * 2;

                // Average 2x2 block
                float sum = 0.0f;
                int count = 0;

                for (int dy = 0; dy < 2 && sy + dy < srcH; dy++)
                {
                    for (int dx = 0; dx < 2 && sx + dx < srcW; dx++)
                    {
                        sum += static_cast<float>(input[(sy + dy) * srcW + (sx + dx)]);
                        count++;
                    }
                }

                output[y * dstW + x] = static_cast<unsigned short>(sum / max(count, 1));
            }

            // =====================================================================
            // Utility: Upsample 2x (bilinear)
            // =====================================================================
            __global__ void Upsample2x(
                const unsigned short *__restrict__ input,
                unsigned short *__restrict__ output,
                int srcW, int srcH, int dstW, int dstH)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                if (x >= dstW || y >= dstH)
                    return;

                // Map to source coordinates
                float srcX = (x - 0.5f) * 0.5f;
                float srcY = (y - 0.5f) * 0.5f;

                int sx0 = max(0, __float2int_rd(srcX));
                int sy0 = max(0, __float2int_rd(srcY));
                int sx1 = min(sx0 + 1, srcW - 1);
                int sy1 = min(sy0 + 1, srcH - 1);

                float fx = srcX - sx0;
                float fy = srcY - sy0;

                float v00 = static_cast<float>(input[sy0 * srcW + sx0]);
                float v10 = static_cast<float>(input[sy0 * srcW + sx1]);
                float v01 = static_cast<float>(input[sy1 * srcW + sx0]);
                float v11 = static_cast<float>(input[sy1 * srcW + sx1]);

                float val = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                            v01 * (1 - fx) * fy + v11 * fx * fy;

                output[y * dstW + x] = static_cast<unsigned short>(fminf(fmaxf(val, 0.0f), 65535.0f));
            }
        }
    }
}
