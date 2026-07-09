#include "pch.h"
#include "HeaderFiles/BriefDescriptorExtractor.h"

#include <algorithm>
#include <cstdint>

namespace WindowsApp::Core
{
    namespace
    {
        // Deterministic pseudo-random pixel-pair offset for BRIEF bit
        // `pairIndex` - fixed across every image/patch (what BRIEF
        // requires). Bit-for-bit identical to
        // WindowsApp::Compute::Kernels::BriefOffset.
        void BriefOffset(int pairIndex, int& dx1, int& dy1, int& dx2, int& dy2)
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
    }

    void ExtractBriefDescriptors(
        const unsigned char* grayImage, int width, int height,
        const Compute::FeaturePoint* points, int numPoints, Compute::BriefDescriptor* outDescriptors)
    {
        if (!grayImage || !points || !outDescriptors) return;

        for (int idx = 0; idx < numPoints; ++idx)
        {
            int cx = static_cast<int>(points[idx].x);
            int cy = static_cast<int>(points[idx].y);

            uint64_t bits[4] = { 0, 0, 0, 0 };

            for (int bit = 0; bit < 256; ++bit)
            {
                int dx1, dy1, dx2, dy2;
                BriefOffset(bit, dx1, dy1, dx2, dy2);

                int x1 = (std::min)((std::max)(cx + dx1, 0), width - 1);
                int y1 = (std::min)((std::max)(cy + dy1, 0), height - 1);
                int x2 = (std::min)((std::max)(cx + dx2, 0), width - 1);
                int y2 = (std::min)((std::max)(cy + dy2, 0), height - 1);

                unsigned char v1 = grayImage[static_cast<size_t>(y1) * width + x1];
                unsigned char v2 = grayImage[static_cast<size_t>(y2) * width + x2];

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
    }
}
