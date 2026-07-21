#pragma once

#include "ComputeTypes.h"
#include <string>
#include <vector>

namespace WindowsApp::Core::Kernels
{
    // Backs CpuComputeBackend::BlockMatchAlign - see IComputeBackend.h's
    // doc comment for the full contract. Plain scalar C++, no SIMD tier
    // split yet (docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md's
    // Architecture note: this starts scalar-only like HomographyMath/
    // LinearSolve/OverlapCulling, earning SIMD tiers only if profiling
    // later shows it's hot, same as this codebase's other kernels did).
    //
    // refData/srcData: RGB48 (unsigned short per channel), width*height*3
    // each, same dimensions. tileSize/searchRadius in pixels.
    // outOffsets: caller-allocated, tilesX*tilesY entries, row-major
    // (tilesX = ceil(width/tileSize), tilesY = ceil(height/tileSize) -
    // caller must compute these consistently with tileSize, this function
    // trusts them rather than recomputing).
    void BlockMatchAlign(
        const unsigned short* refData, const unsigned short* srcData,
        int width, int height, int tileSize, int searchRadius,
        Compute::TileOffset* outOffsets, int tilesX, int tilesY);

    // Hand-rolled JSON, matching BundleAdjustment.h's
    // SerializeCheckpoint/DeserializeCheckpoint convention (no JSON
    // library dependency in this codebase) - used by BurstAlignExecutor to
    // stash a frame's TileOffset field in Task::checkpointJson.
    // Format: {"tilesX":N,"tilesY":M,"offsets":[dx0,dy0,dx1,dy1,...]}
    std::string SerializeTileOffsets(const std::vector<Compute::TileOffset>& offsets, int tilesX, int tilesY);

    // Returns false (leaves outOffsets/outTilesX/outTilesY untouched) on
    // malformed input or an offsets array whose length doesn't match
    // tilesX*tilesY*2.
    bool DeserializeTileOffsets(const std::string& json,
                                 std::vector<Compute::TileOffset>& outOffsets, int& outTilesX, int& outTilesY);
}
