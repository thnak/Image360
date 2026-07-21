#pragma once

#include "ComputeTypes.h"

namespace WindowsApp::Core::Kernels
{
    // Backs CpuComputeBackend::StructureTensorKernelRegression - see
    // IComputeBackend.h's doc comment for the full contract. Super Res
    // Zoom's merge (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.4, docs/
    // superpowers/plans/2026-07-21-superres-structure-tensor-merge.md
    // Task 2): local gradient structure-tensor analysis + anisotropic
    // kernel construction (elongated along edges) + kernel-regression
    // accumulation onto an upsampled grid + noise-based robustness
    // weighting, all combined into one gather-style pass (no atomics,
    // same style as RobustMergeAccumulate/TileFftMerge).
    //
    // frames[0] is the reference (implicit zero offset); frames[1..
    // numFrames) are sampled via perFrameOffsets[k-1] (sub-pixel,
    // tilesX*tilesY entries each, same tile grid as BlockMatchAlign was
    // computed against). output: caller-allocated
    // (width*scaleFactor)*(height*scaleFactor)*3 (RGB48).
    void StructureTensorKernelRegression(
        const unsigned short* const* frames, int numFrames,
        const Compute::TileOffsetF* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        int scaleFactor, float noiseVariance, unsigned short* output);
}
