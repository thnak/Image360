#pragma once

#include <vector>

namespace WindowsApp::Core::Kernels::SeamBlend
{
    // Matches CudaPipeline::MultiBandBlend's existing (dead) default -
    // kept as the same named constant for continuity even though this is
    // an independent, portable reimplementation.
    constexpr int kDefaultNumBands = 5;

    namespace Detail
    {
        // Axis-aligned integer bbox over covered pixels, inclusive. Empty
        // coverage (no covered pixel at all) leaves valid=false.
        struct BBox
        {
            int x0 = 0, y0 = 0, x1 = -1, y1 = -1;
            bool valid = false;
        };

        // Two-pass chamfer distance transform: for every pixel, the
        // (approximate) distance to the nearest pixel where covered[] is
        // false. Uncovered pixels get distance 0. Used as a per-contributor
        // "confidence" - pixels deep inside a contributor's own coverage
        // (far from ITS OWN edges/gaps) are more trustworthy than pixels
        // near an edge, regardless of what other contributors do.
        void ComputeCoverageDistance(const std::vector<unsigned char>& covered,
                                      int width, int height,
                                      std::vector<float>& outDistance);

        // owner[pixel] = argmax_i distances[i][pixel] among contributors
        // with distances[i][pixel] > 0 (i.e. actually covered there - see
        // ComputeCoverageDistance: covered pixels always end up > 0 as long
        // as there's at least one uncovered seed pixel somewhere in that
        // contributor's coverage to measure distance from). -1 if no
        // contributor covers that pixel. Ties -> lowest contributor index,
        // for determinism.
        void ComputeOwnership(const std::vector<std::vector<float>>& distances,
                               int width, int height,
                               std::vector<int>& outOwner);

        BBox ComputeCoverageBBox(const std::vector<unsigned char>& covered, int width, int height);
        bool IntersectBBox(const BBox& a, const BBox& b, BBox& outOverlap);

        // Reroutes the ownership boundary between contributors `i` and `j`
        // within `overlap`, wherever the pixel is currently owned by
        // either of them, via a 1-D dynamic-program seam search along the
        // overlap bbox's dominant axis (row-by-row if wider-than-tall,
        // column-by-column otherwise) minimizing cumulative per-channel
        // color difference between imgI/imgJ crossed by the seam. This is
        // what actively steers the boundary away from a real-world object
        // the two images disagree about (parallax ghosting) instead of
        // always cutting at the geometric midpoint. Only reassigns pixels
        // whose current owner is `i` or `j`; leaves everything else
        // (including pixels some other contributor already owns) alone.
        // A per-axis DP, not full 2-D graph-cut - correct for the common
        // roughly-rectangular overlap between two adjacent images, not for
        // arbitrary irregular overlap shapes.
        void RefineSeamDP(const unsigned short* imgI, const unsigned short* imgJ,
                           const std::vector<float>& distanceI, const std::vector<float>& distanceJ,
                           int width, int height, int contributorI, int contributorJ,
                           const BBox& iBBox, const BBox& jBBox, const BBox& overlap,
                           std::vector<int>& owner);

        // N-way Burt-Adelson multi-band blend given the final per-pixel
        // ownership (owner[pixel] == contributor index, or -1 for no
        // coverage anywhere) and the original per-contributor coverage
        // (distances[i][pixel] > 0). Each contributor's Laplacian pyramid
        // is built from a coverage-normalized ("hole-filled") version of
        // its own warped image - blurring right up to a contributor's own
        // true (0,0,0) edge would otherwise bleed black into the pyramid
        // and show up as dark fringing wherever that contributor wins
        // ownership near its own boundary.
        void MultiBandCompose(const std::vector<const unsigned short*>& warpedBuffers,
                               const std::vector<std::vector<float>>& distances,
                               const std::vector<int>& owner,
                               int width, int height, int numBands,
                               std::vector<unsigned short>& outResult);
    }

    // warpedBuffers: one per chunk contributor, RGB48 (unsigned short per
    // channel), width*height*3 shorts each, row-major, (0,0,0) == "no data"
    // (WarpPerspective's existing convention - see RenderExecutor.cpp).
    // Replaces a naive per-pixel median/average across contributors (which
    // ghosts wherever contributors disagree due to parallax) with:
    // ownership via a per-contributor distance-transform confidence map,
    // seam refinement between overlapping pairs to route the boundary away
    // from high-disagreement content, then a multi-band blend along the
    // resulting seam so the transition itself isn't a visible hard cut.
    void BlendChunkContributors(const std::vector<const unsigned short*>& warpedBuffers,
                                 int width, int height, std::vector<unsigned short>& outResult);
}
