#pragma once

#include "ComputeTypes.h"
#include <string>
#include <vector>

namespace WindowsApp::Core::Kernels
{
    // Sub-pixel refinement of BlockMatchAlign's integer per-tile offsets
    // (docs/superpowers/plans/2026-07-21-superres-structure-tensor-merge.md
    // Task 1, SS2.4 stage (a)) - single-level Lucas-Kanade, host-only (not
    // routed through IComputeBackend, same rationale as HomographyMath/
    // LinearSolve: a tiny per-tile dense system, a GPU round-trip would add
    // latency without benefit).
    //
    // refData/srcData: RGB48, width*height*3 each, same dimensions as the
    // coarseOffsets/outOffsets tile grid was computed against.
    // coarseOffsets/outOffsets: tilesX*tilesY entries, row-major, same
    // convention as BlockMatchAlign's outOffsets. Seeds each tile's
    // Lucas-Kanade iteration from coarseOffsets (already within one pixel
    // of the truth) and writes the sub-pixel-refined result to outOffsets
    // (outOffsets may alias neither buffer). Same sign convention as
    // BlockMatchAlign: src coordinate = ref coordinate + offset. A tile
    // whose local gradients are too weak/degenerate for a well-posed 2x2
    // normal-equations solve keeps its coarse integer offset unrefined
    // (matches BlockMatchAlign's own "no reliable candidate" tolerance),
    // never NaN/divergent.
    void RefineOffsetsSubPixel(
        const unsigned short* refData, const unsigned short* srcData,
        int width, int height, int tileSize,
        const Compute::TileOffset* coarseOffsets, int tilesX, int tilesY,
        int iterations, Compute::TileOffsetF* outOffsets);

    // Hand-rolled JSON, matching SerializeTileOffsets/DeserializeTileOffsets's
    // convention exactly but for float dx/dy (std::stof, not std::stoi).
    // Format: {"tilesX":N,"tilesY":M,"offsets":[dx0,dy0,dx1,dy1,...]}
    std::string SerializeTileOffsetsF(const std::vector<Compute::TileOffsetF>& offsets, int tilesX, int tilesY);

    // Returns false (leaves outOffsets/outTilesX/outTilesY untouched) on
    // malformed input or an offsets array whose length doesn't match
    // tilesX*tilesY*2.
    bool DeserializeTileOffsetsF(const std::string& json,
                                  std::vector<Compute::TileOffsetF>& outOffsets, int& outTilesX, int& outTilesY);
}
