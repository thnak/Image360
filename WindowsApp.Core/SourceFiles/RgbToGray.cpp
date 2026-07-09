#include "pch.h"
#include "HeaderFiles/RgbToGray.h"

#include <algorithm>

namespace WindowsApp::Core
{
    void ConvertRgbToGray(const unsigned char* rgb, int width, int height, unsigned char* outGray)
    {
        if (!rgb || !outGray) return;
        for (int i = 0; i < width * height; ++i)
        {
            const unsigned char* px = rgb + static_cast<size_t>(i) * 3;
            float luma = 0.299f * px[0] + 0.587f * px[1] + 0.114f * px[2];
            outGray[i] = static_cast<unsigned char>((std::min)((std::max)(luma, 0.0f), 255.0f));
        }
    }
}
