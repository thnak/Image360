#include "pch.h"
#include "HeaderFiles/BlockMatchAlignKernel.h"

#include <cstdlib>
#include <limits>
#include <sstream>

namespace WindowsApp::Core::Kernels
{
    namespace
    {
        // SAD between the tile [tx0,tx1) x [ty0,ty1) of refData and the
        // same-sized window of srcData offset by (dx,dy), summed across all
        // 3 RGB48 channels. Returns false (leaves outSad untouched) if any
        // pixel of the offset window would fall outside srcData's bounds -
        // callers skip such candidates rather than treating a partial/
        // clamped window as comparable to a fully-in-bounds one.
        bool TileSad(const unsigned short* refData, const unsigned short* srcData,
                     int width, int height, int tx0, int ty0, int tx1, int ty1,
                     int dx, int dy, int64_t& outSad)
        {
            if (tx0 + dx < 0 || ty0 + dy < 0 || tx1 + dx > width || ty1 + dy > height)
                return false;

            int64_t sad = 0;
            for (int y = ty0; y < ty1; ++y)
            {
                const unsigned short* refRow = refData + (static_cast<size_t>(y) * width) * 3;
                const unsigned short* srcRow = srcData + (static_cast<size_t>(y + dy) * width) * 3;
                for (int x = tx0; x < tx1; ++x)
                {
                    const unsigned short* refPixel = refRow + static_cast<size_t>(x) * 3;
                    const unsigned short* srcPixel = srcRow + static_cast<size_t>(x + dx) * 3;
                    for (int c = 0; c < 3; ++c)
                        sad += std::abs(static_cast<int>(refPixel[c]) - static_cast<int>(srcPixel[c]));
                }
            }
            outSad = sad;
            return true;
        }
    }

    void BlockMatchAlign(
        const unsigned short* refData, const unsigned short* srcData,
        int width, int height, int tileSize, int searchRadius,
        Compute::TileOffset* outOffsets, int tilesX, int tilesY)
    {
        for (int ty = 0; ty < tilesY; ++ty)
        {
            int ty0 = ty * tileSize;
            int ty1 = (std::min)(ty0 + tileSize, height);

            for (int tx = 0; tx < tilesX; ++tx)
            {
                int tx0 = tx * tileSize;
                int tx1 = (std::min)(tx0 + tileSize, width);

                Compute::TileOffset best{ 0, 0 };
                int64_t bestSad = std::numeric_limits<int64_t>::max();
                int bestCost = 0; // |dx|+|dy| of the current best - tie-break toward zero motion

                // dx=dy=0 is always in bounds (the tile's own position),
                // so it's evaluated first and guarantees at least one
                // candidate - every other offset is only ever adopted if
                // it's strictly better.
                for (int dy = -searchRadius; dy <= searchRadius; ++dy)
                {
                    for (int dx = -searchRadius; dx <= searchRadius; ++dx)
                    {
                        int64_t sad;
                        if (!TileSad(refData, srcData, width, height, tx0, ty0, tx1, ty1, dx, dy, sad))
                            continue;

                        int cost = std::abs(dx) + std::abs(dy);
                        if (sad < bestSad || (sad == bestSad && cost < bestCost))
                        {
                            bestSad = sad;
                            bestCost = cost;
                            best = Compute::TileOffset{ dx, dy };
                        }
                    }
                }

                outOffsets[static_cast<size_t>(ty) * tilesX + tx] = best;
            }
        }
    }

    std::string SerializeTileOffsets(const std::vector<Compute::TileOffset>& offsets, int tilesX, int tilesY)
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

    bool DeserializeTileOffsets(const std::string& json,
                                 std::vector<Compute::TileOffset>& outOffsets, int& outTilesX, int& outTilesY)
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

            std::vector<int> flat;
            std::string arrayContent = json.substr(arrayStart, arrayEnd - arrayStart);
            size_t pos = 0;
            while (pos < arrayContent.size())
            {
                size_t comma = arrayContent.find(',', pos);
                std::string token = (comma == std::string::npos)
                    ? arrayContent.substr(pos)
                    : arrayContent.substr(pos, comma - pos);

                if (!token.empty()) flat.push_back(std::stoi(token));

                if (comma == std::string::npos) break;
                pos = comma + 1;
            }

            if (flat.size() != static_cast<size_t>(tilesX) * tilesY * 2) return false;

            std::vector<Compute::TileOffset> offsets;
            offsets.reserve(static_cast<size_t>(tilesX) * tilesY);
            for (size_t i = 0; i < flat.size(); i += 2)
                offsets.push_back(Compute::TileOffset{ flat[i], flat[i + 1] });

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
