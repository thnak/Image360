#include "pch.h"
#include "HeaderFiles/StructureTensorMergeKernel.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace WindowsApp::Compute;

namespace WindowsApp::Core::Kernels
{
    namespace
    {
        // Algorithm-internal tuning constants (not exposed via
        // IComputeBackend - BurstMergeExecutor only varies tileSize/
        // scaleFactor/noiseVariance, same "kernel owns its own constants"
        // precedent as TileFftMergeKernel.cpp's kPi/SampleClamped or
        // SubPixelRefineKernel.cpp's kMaxStepPerIteration).
        constexpr int kStructureTensorRadius = 2;      // 5x5 gradient window
        constexpr float kBaseKernelRadius = 1.0f;      // low-res pixels, isotropic case
        constexpr float kMaxAnisotropy = 2.0f;
        constexpr float kKernelCutoffSigma = 1.5f;     // gather-radius sizing + weight cutoff
        constexpr float kMinKernelWeight = 1e-3f;
        constexpr float kMinWeightSum = 1e-6f;

        inline float SampleClampedF(const unsigned short* frame, int width, int height, int px, int py, int channel)
        {
            int cx = (std::max)(0, (std::min)(width - 1, px));
            int cy = (std::max)(0, (std::min)(height - 1, py));
            return static_cast<float>(frame[(static_cast<size_t>(cy) * width + cx) * 3 + channel]);
        }

        float BilinearSample(const unsigned short* frame, int width, int height, float x, float y, int channel)
        {
            int x0 = static_cast<int>(std::floor(x));
            int y0 = static_cast<int>(std::floor(y));
            float fx = x - static_cast<float>(x0);
            float fy = y - static_cast<float>(y0);

            float v00 = SampleClampedF(frame, width, height, x0, y0, channel);
            float v10 = SampleClampedF(frame, width, height, x0 + 1, y0, channel);
            float v01 = SampleClampedF(frame, width, height, x0, y0 + 1, channel);
            float v11 = SampleClampedF(frame, width, height, x0 + 1, y0 + 1, channel);

            float top = v00 + (v10 - v00) * fx;
            float bottom = v01 + (v11 - v01) * fx;
            return top + (bottom - top) * fy;
        }

        // Per-reference-pixel structure-tensor summary: the dominant
        // gradient direction (cosTheta,sinTheta - the eigenvector of the
        // LARGER eigenvalue, i.e. the direction ACROSS an edge) and a
        // [0,1] coherence measuring how anisotropic the local structure
        // is (0 = flat/isotropic, 1 = a strong, unambiguous edge).
        struct TensorInfo
        {
            float cosTheta = 1.0f;
            float sinTheta = 0.0f;
            float coherence = 0.0f;
        };

        std::vector<TensorInfo> ComputeStructureTensorField(const unsigned short* refData, int width, int height)
        {
            std::vector<float> luma(static_cast<size_t>(width) * height);
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    luma[static_cast<size_t>(y) * width + x] =
                        (SampleClampedF(refData, width, height, x, y, 0)
                         + SampleClampedF(refData, width, height, x, y, 1)
                         + SampleClampedF(refData, width, height, x, y, 2)) / 3.0f;

            auto lumaAt = [&](int x, int y)
            {
                int cx = (std::max)(0, (std::min)(width - 1, x));
                int cy = (std::max)(0, (std::min)(height - 1, y));
                return luma[static_cast<size_t>(cy) * width + cx];
            };

            std::vector<float> gx(static_cast<size_t>(width) * height), gy(static_cast<size_t>(width) * height);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    size_t idx = static_cast<size_t>(y) * width + x;
                    gx[idx] = (lumaAt(x + 1, y) - lumaAt(x - 1, y)) * 0.5f;
                    gy[idx] = (lumaAt(x, y + 1) - lumaAt(x, y - 1)) * 0.5f;
                }
            }

            auto gAt = [&](const std::vector<float>& g, int x, int y)
            {
                int cx = (std::max)(0, (std::min)(width - 1, x));
                int cy = (std::max)(0, (std::min)(height - 1, y));
                return g[static_cast<size_t>(cy) * width + cx];
            };

            std::vector<TensorInfo> field(static_cast<size_t>(width) * height);
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    float sxx = 0.0f, syy = 0.0f, sxy = 0.0f;
                    for (int wy = -kStructureTensorRadius; wy <= kStructureTensorRadius; ++wy)
                    {
                        for (int wx = -kStructureTensorRadius; wx <= kStructureTensorRadius; ++wx)
                        {
                            float ix = gAt(gx, x + wx, y + wy);
                            float iy = gAt(gy, x + wx, y + wy);
                            sxx += ix * ix;
                            syy += iy * iy;
                            sxy += ix * iy;
                        }
                    }

                    float trace = sxx + syy;
                    float half = trace * 0.5f;
                    float disc = (std::max)(0.0f, half * half - (sxx * syy - sxy * sxy));
                    float sq = std::sqrt(disc);
                    float lambda1 = half + sq; // larger
                    float lambda2 = (std::max)(0.0f, half - sq);

                    TensorInfo info;
                    info.coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-6f);
                    float theta = 0.5f * std::atan2(2.0f * sxy, sxx - syy);
                    info.cosTheta = std::cos(theta);
                    info.sinTheta = std::sin(theta);

                    field[static_cast<size_t>(y) * width + x] = info;
                }
            }

            return field;
        }
    }

    void StructureTensorKernelRegression(
        const unsigned short* const* frames, int numFrames,
        const TileOffsetF* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        int scaleFactor, float noiseVariance, unsigned short* output)
    {
        std::vector<TensorInfo> tensorField = ComputeStructureTensorField(frames[0], width, height);

        int outWidth = width * scaleFactor;
        int outHeight = height * scaleFactor;
        const float twoNoiseVariance = 2.0f * noiseVariance;

        for (int oy = 0; oy < outHeight; ++oy)
        {
            for (int ox = 0; ox < outWidth; ++ox)
            {
                float lowResX = (static_cast<float>(ox) + 0.5f) / static_cast<float>(scaleFactor) - 0.5f;
                float lowResY = (static_cast<float>(oy) + 0.5f) / static_cast<float>(scaleFactor) - 0.5f;

                int nearestX = (std::max)(0, (std::min)(width - 1, static_cast<int>(std::lround(lowResX))));
                int nearestY = (std::max)(0, (std::min)(height - 1, static_cast<int>(std::lround(lowResY))));
                const TensorInfo& info = tensorField[static_cast<size_t>(nearestY) * width + nearestX];

                // Gradient direction (across the edge) and its perpendicular
                // (along the edge) - the kernel elongates along the edge
                // (radiusAlong) and compresses across it (radiusAcross).
                float gradDirX = info.cosTheta, gradDirY = info.sinTheta;
                float edgeDirX = -info.sinTheta, edgeDirY = info.cosTheta;
                float anisotropy = 1.0f + (kMaxAnisotropy - 1.0f) * info.coherence;
                float radiusAlong = kBaseKernelRadius * anisotropy;
                float radiusAcross = kBaseKernelRadius / anisotropy;

                int gatherRadius = (std::max)(1, static_cast<int>(
                    std::ceil((std::max)(radiusAlong, radiusAcross) * kKernelCutoffSigma)));

                int tileX = (std::max)(0, (std::min)(tilesX - 1,
                    static_cast<int>(std::floor(lowResX)) / tileSize));
                int tileY = (std::max)(0, (std::min)(tilesY - 1,
                    static_cast<int>(std::floor(lowResY)) / tileSize));

                float refLocal[3];
                for (int c = 0; c < 3; ++c)
                    refLocal[c] = BilinearSample(frames[0], width, height, lowResX, lowResY, c);

                double accum[3] = { 0.0, 0.0, 0.0 };
                double weightSum[3] = { 0.0, 0.0, 0.0 };

                for (int k = 0; k < numFrames; ++k)
                {
                    TileOffsetF offset = (k == 0) ? TileOffsetF{ 0.0f, 0.0f }
                        : perFrameOffsets[k - 1][static_cast<size_t>(tileY) * tilesX + tileX];

                    float fx = lowResX + offset.dx;
                    float fy = lowResY + offset.dy;
                    int baseIx = static_cast<int>(std::floor(fx));
                    int baseIy = static_cast<int>(std::floor(fy));

                    for (int py = baseIy - gatherRadius; py <= baseIy + gatherRadius; ++py)
                    {
                        if (py < 0 || py >= height) continue; // out of bounds - excluded, not zero
                        for (int px = baseIx - gatherRadius; px <= baseIx + gatherRadius; ++px)
                        {
                            if (px < 0 || px >= width) continue;

                            // This raw sample's position, expressed back in
                            // the reference/output coordinate space.
                            float ddx = (static_cast<float>(px) - offset.dx) - lowResX;
                            float ddy = (static_cast<float>(py) - offset.dy) - lowResY;
                            float dAlong = ddx * edgeDirX + ddy * edgeDirY;
                            float dAcross = ddx * gradDirX + ddy * gradDirY;
                            float normAlong = dAlong / radiusAlong;
                            float normAcross = dAcross / radiusAcross;
                            float kernelW = std::exp(-0.5f * (normAlong * normAlong + normAcross * normAcross));
                            if (kernelW < kMinKernelWeight) continue;

                            size_t pixelIdx = (static_cast<size_t>(py) * width + px) * 3;
                            for (int c = 0; c < 3; ++c)
                            {
                                float sampleVal = static_cast<float>(frames[k][pixelIdx + c]);
                                float delta = sampleVal - refLocal[c];
                                float robustW = std::exp(-(delta * delta) / twoNoiseVariance);
                                float w = kernelW * robustW;

                                accum[c] += static_cast<double>(w) * sampleVal;
                                weightSum[c] += w;
                            }
                        }
                    }
                }

                size_t outIdx = (static_cast<size_t>(oy) * outWidth + ox) * 3;
                for (int c = 0; c < 3; ++c)
                {
                    float merged = (weightSum[c] > kMinWeightSum)
                        ? static_cast<float>(accum[c] / weightSum[c])
                        : refLocal[c]; // negligible support - fall back to the reference, never a spurious zero
                    merged = (std::max)(0.0f, (std::min)(65535.0f, merged));
                    output[outIdx + c] = static_cast<unsigned short>(merged + 0.5f);
                }
            }
        }
    }
}
