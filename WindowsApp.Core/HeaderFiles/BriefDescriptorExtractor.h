#pragma once

#include "ComputeTypes.h"

namespace WindowsApp::Core
{
    // 256-bit BRIEF descriptor extraction on a grayscale image (see
    // RgbToGray.h) - replicates
    // WindowsApp::Compute::Kernels::BriefDescribeKernel's deterministic
    // pseudo-random sampling pattern exactly. grayImage: width*height, one
    // byte/pixel - same buffer DetectFastCorners was run on. points:
    // FeaturePoint list (e.g. from DetectFastCorners). outDescriptors:
    // caller-allocated, numPoints capacity.
    void ExtractBriefDescriptors(
        const unsigned char* grayImage, int width, int height,
        const Compute::FeaturePoint* points, int numPoints, Compute::BriefDescriptor* outDescriptors);
}
