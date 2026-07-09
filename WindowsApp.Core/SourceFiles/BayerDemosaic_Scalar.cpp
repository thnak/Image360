#include "pch.h"
#include "HeaderFiles/BayerDemosaicKernels.h"

#include <algorithm>
#include <vector>

namespace WindowsApp::Core::Kernels::Scalar
{
    void DemosaicBayer(const unsigned short* cfaData, int width, int height,
                        unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
                        uint32_t filters, unsigned short* rgbOut)
    {
        int numPixels = width * height;
        std::vector<unsigned short> cfa(cfaData, cfaData + numPixels);

        // Black level subtract
        for (int i = 0; i < numPixels; ++i)
        {
            cfa[i] = (cfa[i] > blackLevel) ? (cfa[i] - blackLevel) : 0;
        }

        // White balance
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                int idx = y * width + x;
                int channel = Detail::BayerColor(y, x, filters);
                float gain = camMul[channel];
                float val = static_cast<float>(cfa[idx]) * gain;
                cfa[idx] = static_cast<unsigned short>((std::min)((std::max)(val, 0.0f), 65535.0f));
            }
        }

        // Bilinear demosaic + color matrix
        std::vector<unsigned short> demosaiced(static_cast<size_t>(numPixels) * 3);
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Detail::DemosaicPixel(cfa.data(), width, height, x, y, filters,
                                       demosaiced.data() + (static_cast<size_t>(y) * width + x) * 3);
            }
        }

        for (int i = 0; i < numPixels; ++i)
        {
            Detail::ApplyColorMatrixPixel(demosaiced.data() + static_cast<size_t>(i) * 3, rgbCam, rgbOut + static_cast<size_t>(i) * 3);
        }
    }
}
