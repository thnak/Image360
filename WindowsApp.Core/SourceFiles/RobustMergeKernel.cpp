#include "pch.h"
#include "HeaderFiles/RobustMergeKernel.h"

#include <algorithm>
#include <cmath>

namespace WindowsApp::Core::Kernels
{
    void RobustMergeAccumulate(
        const unsigned short* const* frames, int numFrames,
        const Compute::TileOffset* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        float sigma, unsigned short* output)
    {
        const unsigned short* reference = frames[0];
        const float twoSigmaSq = 2.0f * sigma * sigma;

        for (int y = 0; y < height; ++y)
        {
            int tileY = (std::min)(y / tileSize, tilesY - 1);
            for (int x = 0; x < width; ++x)
            {
                int tileX = (std::min)(x / tileSize, tilesX - 1);
                size_t pixelIndex = (static_cast<size_t>(y) * width + x) * 3;

                for (int c = 0; c < 3; ++c)
                {
                    float refSample = static_cast<float>(reference[pixelIndex + c]);

                    // Reference always contributes at weight 1.0 - the
                    // anchor every other frame is compared against, per
                    // this kernel's doc comment.
                    float weightedSum = refSample;
                    float weightSum = 1.0f;

                    for (int k = 1; k < numFrames; ++k)
                    {
                        const Compute::TileOffset& offset = perFrameOffsets[k - 1][static_cast<size_t>(tileY) * tilesX + tileX];
                        int sx = x + offset.dx;
                        int sy = y + offset.dy;
                        if (sx < 0 || sy < 0 || sx >= width || sy >= height)
                            continue; // out of bounds after alignment - excluded, not treated as zero

                        float sample = static_cast<float>(
                            frames[k][(static_cast<size_t>(sy) * width + sx) * 3 + c]);
                        float delta = sample - refSample;
                        float weight = std::exp(-(delta * delta) / twoSigmaSq);

                        weightedSum += weight * sample;
                        weightSum += weight;
                    }

                    float merged = weightedSum / weightSum;
                    output[pixelIndex + c] = static_cast<unsigned short>(
                        (std::min)((std::max)(merged, 0.0f), 65535.0f));
                }
            }
        }
    }
}
