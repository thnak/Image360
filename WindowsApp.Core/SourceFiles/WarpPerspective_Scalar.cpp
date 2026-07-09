#include "pch.h"
#include "HeaderFiles/WarpPerspectiveKernels.h"

#include <cmath>

namespace WindowsApp::Core::Kernels::Scalar
{
    void WarpPerspective(const unsigned short* srcData, int srcW, int srcH,
                          unsigned short* dstData, int dstW, int dstH,
                          const float* invH, int offsetX, int offsetY)
    {
        for (int y = 0; y < dstH; ++y)
        {
            float dstY = static_cast<float>(y + offsetY);
            unsigned short* dstRow = dstData + static_cast<size_t>(y) * dstW * 3;

            for (int x = 0; x < dstW; ++x)
            {
                float dstX = static_cast<float>(x + offsetX);

                float srcX = invH[0] * dstX + invH[1] * dstY + invH[2];
                float srcY = invH[3] * dstX + invH[4] * dstY + invH[5];
                float srcW1 = invH[6] * dstX + invH[7] * dstY + invH[8];

                unsigned short* dstPixel = dstRow + static_cast<size_t>(x) * 3;

                if (std::fabs(srcW1) < 1e-10f)
                {
                    dstPixel[0] = dstPixel[1] = dstPixel[2] = 0;
                    continue;
                }

                srcX /= srcW1;
                srcY /= srcW1;

                int sx0 = static_cast<int>(std::floor(srcX));
                int sy0 = static_cast<int>(std::floor(srcY));
                float fx = srcX - sx0;
                float fy = srcY - sy0;

                bool valid = (sx0 >= 0 && sy0 >= 0 && sx0 + 1 < srcW && sy0 + 1 < srcH);
                Detail::SampleBilinear(srcData, srcW, srcH, dstPixel, sx0, sy0, fx, fy, valid);
            }
        }
    }
}
