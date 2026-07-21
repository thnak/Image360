#pragma once

#include "ComputeTypes.h"

namespace WindowsApp::Core::Kernels
{
    // Backs CpuComputeBackend::RobustMergeAccumulate - see
    // IComputeBackend.h's doc comment for the full contract. Plain scalar
    // C++, no SIMD tier split yet (same rationale as BlockMatchAlignKernel.h).
    void RobustMergeAccumulate(
        const unsigned short* const* frames, int numFrames,
        const Compute::TileOffset* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        float sigma, unsigned short* output);
}
