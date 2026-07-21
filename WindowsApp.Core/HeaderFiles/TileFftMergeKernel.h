#pragma once

#include "ComputeTypes.h"

namespace WindowsApp::Core::Kernels
{
    // Backs CpuComputeBackend::TileFftMerge - see IComputeBackend.h's doc
    // comment for the full contract, and docs/superpowers/plans/
    // 2026-07-21-hdrplus-tile-fft-merge.md for the algorithm. tileSize
    // MUST be a power of two (the FFT is radix-2) - returns false (leaves
    // output untouched) otherwise; the caller (CpuComputeBackend) maps
    // that to ComputeResult::INVALID_PARAM.
    bool TileFftMerge(
        const unsigned short* const* frames, int numFrames,
        const Compute::TileOffset* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        float noiseVariance, unsigned short* output);
}
