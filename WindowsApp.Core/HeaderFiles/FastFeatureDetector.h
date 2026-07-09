#pragma once

#include "ComputeTypes.h"

namespace WindowsApp::Core
{
    // FAST-9 corner detection on a grayscale image (see RgbToGray.h for the
    // RGB8->gray conversion) - replicates
    // WindowsApp::Compute::Kernels::FastDetectKernel's algorithm exactly
    // (16-point Bresenham circle, threshold 20, 3px margin, >=9-in-a-row
    // brighter/darker run) so CPU and GPU backends agree on what a
    // detected corner is. grayImage: width*height, one byte/pixel.
    // outPoints: caller-allocated, maxPoints capacity. outCount: actual
    // detections (<= maxPoints), may exceed maxPoints if more were found
    // (matching the GPU kernel's own atomicAdd-then-clamp behavior).
    void DetectFastCorners(
        const unsigned char* grayImage, int width, int height,
        Compute::FeaturePoint* outPoints, int* outCount, int maxPoints);
}
