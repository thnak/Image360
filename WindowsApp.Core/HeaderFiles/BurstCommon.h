#pragma once
#include "ProjectManager.h"
#include <limits>
#include <vector>

namespace WindowsApp::Core
{
    // Shared by BurstAlignExecutor and BurstMergeExecutor - docs/
    // superpowers/plans/2026-07-21-mfnr-block-match-merge.md Tasks 4-5.
    // Kept in one place rather than duplicated per-file so the two
    // executors can never silently drift apart on tile geometry.

    // Picked as "a real, testable default," not tuned against real photos
    // yet.
    inline constexpr int kBurstTileSize = 16;
    inline constexpr int kBurstSearchRadius = 8;

    // RobustMergeAccumulate's Gaussian-weight noise parameter - also not
    // calibrated per-ISO (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.3's
    // correction: that's HDR+'s real noise model, tracked follow-up here).
    inline constexpr float kBurstMergeSigma = 2000.0f;

    // The lowest-id frame (the first one added) is the alignment/merge
    // reference - not necessarily images.front(), since GetInputImages()'s
    // ordering isn't part of its documented contract.
    inline int BurstReferenceImageId(const std::vector<InputImageModel>& images)
    {
        int minId = (std::numeric_limits<int>::max)();
        for (const auto& img : images) minId = (std::min)(minId, img.id);
        return minId;
    }
}
