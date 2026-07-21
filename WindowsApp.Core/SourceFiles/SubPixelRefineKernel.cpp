#include "pch.h"
#include "HeaderFiles/SubPixelRefineKernel.h"

#include <cmath>
#include <sstream>

using namespace WindowsApp::Compute;

namespace WindowsApp::Core::Kernels
{
    namespace
    {
        // Per-iteration step clamp - keeps a low-texture/degenerate tile's
        // solve from diverging wildly on one bad step (same defensive
        // intent as BlockMatchAlign's exhaustive-search bound, just for a
        // gradient-descent-style update instead).
        constexpr float kMaxStepPerIteration = 1.0f;
        // Below this determinant, the tile's local gradients don't
        // constrain a 2x2 solve well enough to trust (flat/degenerate
        // tile) - refinement stops early and the coarse integer offset is
        // kept as-is.
        constexpr float kMinDeterminant = 1e-3f;

        inline float SampleClampedF(const unsigned short* frame, int width, int height, int px, int py, int channel)
        {
            int cx = (std::max)(0, (std::min)(width - 1, px));
            int cy = (std::max)(0, (std::min)(height - 1, py));
            return static_cast<float>(frame[(static_cast<size_t>(cy) * width + cx) * 3 + channel]);
        }

        // Clamp-to-edge bilinear sample - never reads out of bounds, never
        // zero-pads (same "gap != black" principle used throughout this
        // codebase's other kernels).
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
    }

    void RefineOffsetsSubPixel(
        const unsigned short* refData, const unsigned short* srcData,
        int width, int height, int tileSize,
        const TileOffset* coarseOffsets, int tilesX, int tilesY,
        int iterations, TileOffsetF* outOffsets)
    {
        for (int ty = 0; ty < tilesY; ++ty)
        {
            int ty0 = ty * tileSize;
            int ty1 = (std::min)(ty0 + tileSize, height);

            for (int tx = 0; tx < tilesX; ++tx)
            {
                int tx0 = tx * tileSize;
                int tx1 = (std::min)(tx0 + tileSize, width);

                size_t idx = static_cast<size_t>(ty) * tilesX + tx;
                float dx = static_cast<float>(coarseOffsets[idx].dx);
                float dy = static_cast<float>(coarseOffsets[idx].dy);

                for (int iter = 0; iter < iterations; ++iter)
                {
                    double sumIxIx = 0.0, sumIyIy = 0.0, sumIxIy = 0.0;
                    double sumIxIt = 0.0, sumIyIt = 0.0;

                    for (int y = ty0; y < ty1; ++y)
                    {
                        for (int x = tx0; x < tx1; ++x)
                        {
                            for (int c = 0; c < 3; ++c)
                            {
                                // Reference gradient, central difference
                                // (clamped) - independent of dx/dy, but
                                // recomputed each iteration for simplicity
                                // over a 16x16 tile rather than caching
                                // across iterations (this op is not hot
                                // enough yet to warrant it, same "earn a
                                // SIMD/caching tier if profiling shows it's
                                // needed" stance as BlockMatchAlignKernel.h).
                                float ix = (SampleClampedF(refData, width, height, x + 1, y, c)
                                            - SampleClampedF(refData, width, height, x - 1, y, c)) * 0.5f;
                                float iy = (SampleClampedF(refData, width, height, x, y + 1, c)
                                            - SampleClampedF(refData, width, height, x, y - 1, c)) * 0.5f;

                                float refVal = SampleClampedF(refData, width, height, x, y, c);
                                float warpedVal = BilinearSample(srcData, width, height,
                                                                  static_cast<float>(x) + dx,
                                                                  static_cast<float>(y) + dy, c);
                                float it = warpedVal - refVal;

                                sumIxIx += static_cast<double>(ix) * ix;
                                sumIyIy += static_cast<double>(iy) * iy;
                                sumIxIy += static_cast<double>(ix) * iy;
                                sumIxIt += static_cast<double>(ix) * it;
                                sumIyIt += static_cast<double>(iy) * it;
                            }
                        }
                    }

                    double det = sumIxIx * sumIyIy - sumIxIy * sumIxIy;
                    if (std::fabs(det) < kMinDeterminant) break; // degenerate tile - keep current estimate

                    // Solve [[sumIxIx,sumIxIy],[sumIxIy,sumIyIy]] * delta = -[sumIxIt,sumIyIt]
                    double stepDx = (-sumIxIt * sumIyIy + sumIyIt * sumIxIy) / det;
                    double stepDy = (-sumIyIt * sumIxIx + sumIxIt * sumIxIy) / det;

                    stepDx = (std::max)(-static_cast<double>(kMaxStepPerIteration),
                                         (std::min)(static_cast<double>(kMaxStepPerIteration), stepDx));
                    stepDy = (std::max)(-static_cast<double>(kMaxStepPerIteration),
                                         (std::min)(static_cast<double>(kMaxStepPerIteration), stepDy));

                    dx += static_cast<float>(stepDx);
                    dy += static_cast<float>(stepDy);
                }

                outOffsets[idx] = TileOffsetF{ dx, dy };
            }
        }

        // Each tile's LK estimate is a small (tileSize x tileSize) sample
        // under real sensor/quantization noise - independently, tile-to-
        // tile scatter is large enough (empirically verified while writing
        // this phase's e2e test - see docs/superpowers/plans/
        // 2026-07-21-superres-structure-tensor-merge.md SS9) to create
        // visible tile-boundary artifacts once StructureTensorKernelRegression
        // treats each tile as an independent constant-offset region. A
        // light 3x3-neighborhood smoothing pass over the offset FIELD
        // (not a wider per-tile sample - that would need re-running the LK
        // solve, this just regularizes the already-computed field) trades
        // a little true local motion resolution for a much more reliable
        // field, the same motion-vector-field regularization tradeoff
        // video coding/optical flow pipelines make routinely.
        if (tilesX * tilesY > 1)
        {
            std::vector<TileOffsetF> smoothed(static_cast<size_t>(tilesX) * tilesY);
            for (int ty = 0; ty < tilesY; ++ty)
            {
                for (int tx = 0; tx < tilesX; ++tx)
                {
                    float sumDx = 0.0f, sumDy = 0.0f;
                    int count = 0;
                    for (int ny = (std::max)(0, ty - 1); ny <= (std::min)(tilesY - 1, ty + 1); ++ny)
                    {
                        for (int nx = (std::max)(0, tx - 1); nx <= (std::min)(tilesX - 1, tx + 1); ++nx)
                        {
                            const TileOffsetF& o = outOffsets[static_cast<size_t>(ny) * tilesX + nx];
                            sumDx += o.dx;
                            sumDy += o.dy;
                            ++count;
                        }
                    }
                    smoothed[static_cast<size_t>(ty) * tilesX + tx] = TileOffsetF{ sumDx / count, sumDy / count };
                }
            }
            std::copy(smoothed.begin(), smoothed.end(), outOffsets);
        }
    }

    std::string SerializeTileOffsetsF(const std::vector<TileOffsetF>& offsets, int tilesX, int tilesY)
    {
        std::ostringstream oss;
        oss << "{\"tilesX\":" << tilesX << ",\"tilesY\":" << tilesY << ",\"offsets\":[";
        for (size_t i = 0; i < offsets.size(); ++i)
        {
            if (i > 0) oss << ",";
            oss << offsets[i].dx << "," << offsets[i].dy;
        }
        oss << "]}";
        return oss.str();
    }

    bool DeserializeTileOffsetsF(const std::string& json,
                                  std::vector<TileOffsetF>& outOffsets, int& outTilesX, int& outTilesY)
    {
        const std::string tilesXKey = "\"tilesX\":";
        const std::string tilesYKey = "\"tilesY\":";
        const std::string offsetsKey = "\"offsets\":[";

        size_t tilesXPos = json.find(tilesXKey);
        size_t tilesYPos = json.find(tilesYKey);
        size_t offsetsPos = json.find(offsetsKey);
        if (tilesXPos == std::string::npos || tilesYPos == std::string::npos || offsetsPos == std::string::npos)
            return false;

        try
        {
            int tilesX = std::stoi(json.substr(tilesXPos + tilesXKey.size()));
            int tilesY = std::stoi(json.substr(tilesYPos + tilesYKey.size()));

            size_t arrayStart = offsetsPos + offsetsKey.size();
            size_t arrayEnd = json.find(']', arrayStart);
            if (arrayEnd == std::string::npos) return false;

            std::vector<float> flat;
            std::string arrayContent = json.substr(arrayStart, arrayEnd - arrayStart);
            size_t pos = 0;
            while (pos < arrayContent.size())
            {
                size_t comma = arrayContent.find(',', pos);
                std::string token = (comma == std::string::npos)
                    ? arrayContent.substr(pos)
                    : arrayContent.substr(pos, comma - pos);

                if (!token.empty()) flat.push_back(std::stof(token));

                if (comma == std::string::npos) break;
                pos = comma + 1;
            }

            if (flat.size() != static_cast<size_t>(tilesX) * tilesY * 2) return false;

            std::vector<TileOffsetF> offsets;
            offsets.reserve(static_cast<size_t>(tilesX) * tilesY);
            for (size_t i = 0; i < flat.size(); i += 2)
                offsets.push_back(TileOffsetF{ flat[i], flat[i + 1] });

            outOffsets = std::move(offsets);
            outTilesX = tilesX;
            outTilesY = tilesY;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
}
