#include "pch.h"
#include "HeaderFiles/OverlapCulling.h"
#include "HeaderFiles/ImageLoader.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace WindowsApp::Core
{
    bool ImageOverlapsChunk(const Homography& h, int imageWidth, int imageHeight, const ChunkModel& chunk)
    {
        float corners[4][2] = {
            { 0.0f, 0.0f },
            { static_cast<float>(imageWidth), 0.0f },
            { 0.0f, static_cast<float>(imageHeight) },
            { static_cast<float>(imageWidth), static_cast<float>(imageHeight) }
        };

        float minX = (std::numeric_limits<float>::max)();
        float minY = (std::numeric_limits<float>::max)();
        float maxX = -(std::numeric_limits<float>::max)();
        float maxY = -(std::numeric_limits<float>::max)();

        const auto& m = h.h;
        for (const auto& corner : corners)
        {
            float denom = m[6] * corner[0] + m[7] * corner[1] + m[8];
            if (std::fabs(denom) < 1e-10f) denom = (denom >= 0.0f) ? 1e-10f : -1e-10f;

            float px = (m[0] * corner[0] + m[1] * corner[1] + m[2]) / denom;
            float py = (m[3] * corner[0] + m[4] * corner[1] + m[5]) / denom;

            minX = (std::min)(minX, px);
            minY = (std::min)(minY, py);
            maxX = (std::max)(maxX, px);
            maxY = (std::max)(maxY, py);
        }

        float chunkMinX = static_cast<float>(chunk.x_offset);
        float chunkMinY = static_cast<float>(chunk.y_offset);
        float chunkMaxX = chunkMinX + static_cast<float>(chunk.width);
        float chunkMaxY = chunkMinY + static_cast<float>(chunk.height);

        return (minX < chunkMaxX) && (maxX > chunkMinX) && (minY < chunkMaxY) && (maxY > chunkMinY);
    }

    std::vector<int> FindOverlappingImages(const ChunkModel& chunk, const std::vector<InputImageModel>& images)
    {
        std::vector<int> result;

        for (const auto& image : images)
        {
            int width = 0;
            int height = 0;

            if (image.cfaType == CfaType::STANDARD_RGB)
            {
                GetStandardImageDimensions(image.file_path, width, height);
            }
            else
            {
                ImageLoader loader;
                if (loader.Open(image.file_path))
                {
                    ImageMetadata metadata;
                    if (loader.GetMetadata(metadata))
                    {
                        width = metadata.width;
                        height = metadata.height;
                    }
                }
            }

            if (width <= 0 || height <= 0)
            {
                // Can't determine this image's footprint - conservative
                // default is "overlaps" (see header comment).
                result.push_back(image.id);
                continue;
            }

            if (ImageOverlapsChunk(image.homography, width, height, chunk))
            {
                result.push_back(image.id);
            }
        }

        return result;
    }
}
