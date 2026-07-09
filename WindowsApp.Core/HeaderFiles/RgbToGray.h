#pragma once

namespace WindowsApp::Core
{
    // Matches WindowsApp::Compute::Kernels::RgbToGrayKernel's luma formula
    // exactly (0.299/0.587/0.114 weights, clamped to [0,255]) so CPU
    // feature detection agrees with what the GPU backend would have seen.
    // rgb: interleaved RGB8, width*height*3. outGray: caller-allocated,
    // width*height capacity.
    void ConvertRgbToGray(const unsigned char* rgb, int width, int height, unsigned char* outGray);
}
