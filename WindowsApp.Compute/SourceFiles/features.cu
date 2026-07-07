#include "HeaderFiles/features.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <climits>

namespace WindowsApp
{
    namespace Compute
    {
        namespace Kernels
        {
            __constant__ int kFastDx[16] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
            __constant__ int kFastDy[16] = { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 };

            __device__ __forceinline__ bool HasContiguousRun(const bool flags[16], int minRun)
            {
                int maxRun = 0;
                int curRun = 0;
                // Two passes around the circle so a run that wraps past
                // index 15 back to 0 is still counted correctly.
                for (int i = 0; i < 32; ++i)
                {
                    if (flags[i % 16])
                    {
                        curRun++;
                        if (curRun > maxRun) maxRun = curRun;
                    }
                    else
                    {
                        curRun = 0;
                    }
                }
                return maxRun >= minRun;
            }

            // Deterministic pseudo-random pixel-pair offset for BRIEF bit
            // `pairIndex` - fixed across every image/patch (what BRIEF
            // requires), not sourced from any particular published table.
            __device__ __forceinline__ void BriefOffset(int pairIndex, int& dx1, int& dy1, int& dx2, int& dy2)
            {
                unsigned int h = static_cast<unsigned int>(pairIndex) * 2654435761u;
                h ^= h >> 13; h *= 60493u; h ^= h >> 15;
                dx1 = static_cast<int>(h % 31) - 15;

                h = h * 2246822519u + 1u;
                h ^= h >> 13; h *= 60493u; h ^= h >> 15;
                dy1 = static_cast<int>(h % 31) - 15;

                h = h * 3266489917u + 1u;
                h ^= h >> 13; h *= 60493u; h ^= h >> 15;
                dx2 = static_cast<int>(h % 31) - 15;

                h = h * 668265263u + 1u;
                h ^= h >> 13; h *= 60493u; h ^= h >> 15;
                dy2 = static_cast<int>(h % 31) - 15;
            }

            __global__ void RgbToGrayKernel(
                const unsigned char* __restrict__ rgb,
                unsigned char* __restrict__ gray,
                int width, int height)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;
                if (x >= width || y >= height)
                    return;

                int idx = y * width + x;
                int base = idx * 3;
                float luma = 0.299f * rgb[base] + 0.587f * rgb[base + 1] + 0.114f * rgb[base + 2];
                gray[idx] = static_cast<unsigned char>(fminf(fmaxf(luma, 0.0f), 255.0f));
            }

            __global__ void FastDetectKernel(
                const unsigned char* __restrict__ grayImage, int width, int height,
                FeaturePoint* __restrict__ outPoints, int* __restrict__ outCount, int maxPoints,
                unsigned char threshold)
            {
                int x = blockIdx.x * blockDim.x + threadIdx.x;
                int y = blockIdx.y * blockDim.y + threadIdx.y;

                const int margin = 3;
                if (x < margin || y < margin || x >= width - margin || y >= height - margin)
                    return;

                int centerVal = grayImage[y * width + x];
                int brighterThresh = centerVal + threshold;
                int darkerThresh = centerVal - threshold;

                bool brighter[16];
                bool darker[16];
                for (int i = 0; i < 16; ++i)
                {
                    int nx = x + kFastDx[i];
                    int ny = y + kFastDy[i];
                    int val = grayImage[ny * width + nx];
                    brighter[i] = val > brighterThresh;
                    darker[i] = val < darkerThresh;
                }

                if (!HasContiguousRun(brighter, 9) && !HasContiguousRun(darker, 9))
                    return;

                int idx = atomicAdd(outCount, 1);
                if (idx < maxPoints)
                {
                    outPoints[idx].x = static_cast<float>(x);
                    outPoints[idx].y = static_cast<float>(y);
                }
            }

            __global__ void BriefDescribeKernel(
                const unsigned char* __restrict__ grayImage, int width, int height,
                const FeaturePoint* __restrict__ points, int numPoints,
                BriefDescriptor* __restrict__ outDescriptors)
            {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx >= numPoints)
                    return;

                int cx = static_cast<int>(points[idx].x);
                int cy = static_cast<int>(points[idx].y);

                uint64_t bits[4] = { 0, 0, 0, 0 };

                for (int bit = 0; bit < 256; ++bit)
                {
                    int dx1, dy1, dx2, dy2;
                    BriefOffset(bit, dx1, dy1, dx2, dy2);

                    int x1 = min(max(cx + dx1, 0), width - 1);
                    int y1 = min(max(cy + dy1, 0), height - 1);
                    int x2 = min(max(cx + dx2, 0), width - 1);
                    int y2 = min(max(cy + dy2, 0), height - 1);

                    unsigned char v1 = grayImage[y1 * width + x1];
                    unsigned char v2 = grayImage[y2 * width + x2];

                    if (v1 < v2)
                    {
                        bits[bit / 64] |= (static_cast<uint64_t>(1) << (bit % 64));
                    }
                }

                outDescriptors[idx][0] = bits[0];
                outDescriptors[idx][1] = bits[1];
                outDescriptors[idx][2] = bits[2];
                outDescriptors[idx][3] = bits[3];
            }

            __device__ __forceinline__ int HammingDistance256(const BriefDescriptor& a, const BriefDescriptor& b)
            {
                return __popcll(a[0] ^ b[0]) + __popcll(a[1] ^ b[1])
                     + __popcll(a[2] ^ b[2]) + __popcll(a[3] ^ b[3]);
            }

            __global__ void BruteForceMatchKernel(
                const BriefDescriptor* __restrict__ descA, int countA,
                const BriefDescriptor* __restrict__ descB, int countB,
                MatchResult* __restrict__ outMatches, int* __restrict__ outMatchCount, int maxMatches,
                float ratioThreshold)
            {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i >= countA)
                    return;

                int best = INT_MAX;
                int second = INT_MAX;
                int bestIdx = -1;

                for (int j = 0; j < countB; ++j)
                {
                    int dist = HammingDistance256(descA[i], descB[j]);
                    if (dist < best)
                    {
                        second = best;
                        best = dist;
                        bestIdx = j;
                    }
                    else if (dist < second)
                    {
                        second = dist;
                    }
                }

                if (bestIdx < 0)
                    return;

                bool accept = (second == INT_MAX)
                    || (static_cast<float>(best) < ratioThreshold * static_cast<float>(second));
                if (!accept)
                    return;

                int idx = atomicAdd(outMatchCount, 1);
                if (idx < maxMatches)
                {
                    outMatches[idx].indexA = i;
                    outMatches[idx].indexB = bestIdx;
                    outMatches[idx].hammingDistance = best;
                }
            }
        }
    }
}
