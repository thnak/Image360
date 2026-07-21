#pragma once

#include <vector>

namespace WindowsApp::Core::Kernels::ExposureFusion
{
    // Matches SeamBlendKernels.h's kDefaultNumBands - not shared as a
    // literal constant across the two headers since they're deliberately
    // independent (see FuseTwoExposures's own comment below).
    constexpr int kDefaultNumBands = 5;

    // Mertens-style (well-exposedness-only, docs/superpowers/plans/
    // 2026-07-21-hdrplus-tile-fft-merge.md SS9) two-image Laplacian-
    // pyramid blend. low/high: RGB48, width*height*3, same dimensions -
    // typically two synthetic tone-curve renderings of the SAME merged
    // image (BurstMergeExecutor's HDR_PLUS finish path), not independently
    // captured brackets, so there's no ghosting/alignment concern and no
    // need for SeamBlendKernels.h's ownership/seam-routing/hole-filling
    // machinery (that exists specifically for panorama's partial-coverage,
    // possibly-disagreeing warped buffers) - both inputs here are dense,
    // full-frame, and always agree on content. A small, independent
    // reimplementation of the same underlying Gaussian/Laplacian
    // build+blend+reconstruct math SeamBlend.cpp already proved, not a
    // reuse of its (differently-shaped) public API. outResult: resized
    // internally to width*height*3.
    void FuseTwoExposures(
        const unsigned short* low, const unsigned short* high,
        int width, int height, int numBands,
        std::vector<unsigned short>& outResult);
}
