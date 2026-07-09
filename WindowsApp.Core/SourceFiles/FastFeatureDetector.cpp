#include "pch.h"
#include "HeaderFiles/FastFeatureDetector.h"

namespace WindowsApp::Core
{
    namespace
    {
        constexpr int kFastDx[16] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
        constexpr int kFastDy[16] = { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 };
        constexpr unsigned char kFastThreshold = 20;
        constexpr int kMargin = 3;
        constexpr int kMinRun = 9;

        bool HasContiguousRun(const bool flags[16], int minRun)
        {
            int maxRun = 0;
            int curRun = 0;
            // Two passes around the circle so a run that wraps past index
            // 15 back to 0 is still counted correctly.
            for (int i = 0; i < 32; ++i)
            {
                if (flags[i % 16])
                {
                    ++curRun;
                    if (curRun > maxRun) maxRun = curRun;
                }
                else
                {
                    curRun = 0;
                }
            }
            return maxRun >= minRun;
        }
    }

    void DetectFastCorners(
        const unsigned char* grayImage, int width, int height,
        Compute::FeaturePoint* outPoints, int* outCount, int maxPoints)
    {
        *outCount = 0;
        if (!grayImage || !outPoints || !outCount || width <= 2 * kMargin || height <= 2 * kMargin) return;

        int count = 0;
        for (int y = kMargin; y < height - kMargin; ++y)
        {
            for (int x = kMargin; x < width - kMargin; ++x)
            {
                int centerVal = grayImage[static_cast<size_t>(y) * width + x];
                int brighterThresh = centerVal + kFastThreshold;
                int darkerThresh = centerVal - kFastThreshold;

                bool brighter[16];
                bool darker[16];
                for (int i = 0; i < 16; ++i)
                {
                    int nx = x + kFastDx[i];
                    int ny = y + kFastDy[i];
                    int val = grayImage[static_cast<size_t>(ny) * width + nx];
                    brighter[i] = val > brighterThresh;
                    darker[i] = val < darkerThresh;
                }

                if (!HasContiguousRun(brighter, kMinRun) && !HasContiguousRun(darker, kMinRun))
                    continue;

                if (count < maxPoints)
                {
                    outPoints[count].x = static_cast<float>(x);
                    outPoints[count].y = static_cast<float>(y);
                }
                ++count;
            }
        }

        *outCount = count;
    }
}
