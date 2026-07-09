#include "pch.h"
#include "HeaderFiles/RenderExecutor.h"
#include "HeaderFiles/HomographyMath.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        // A generous sanity cap on contributors per chunk - no longer tied
        // to a kernel-imposed limit (combining is now the per-pixel
        // CombineIgnoringGaps below, not IComputeBackend::MedianStack).
        constexpr size_t kMaxContributorsForMedianStack = 32;

        std::optional<int64_t> FindColorBlobId(ProjectManager& projectManager, int imageId)
        {
            std::string key = std::to_string(imageId);
            for (const auto& task : projectManager.GetTasksForStage(PipelineStage::STAGE2_OPTIMIZE))
            {
                if (task.unitKind == "color" && task.unitKey == key && task.status == TaskStatus::COMPLETED)
                {
                    return task.outputBlobId;
                }
            }
            return std::nullopt;
        }

        // IComputeBackend::MedianStack has no notion of "this contributor
        // doesn't cover this pixel" - it just sees numbers, and
        // WarpPerspective correctly writes (0,0,0) for destination pixels
        // outside the source image's bounds. Chunk contributors usually
        // only pairwise-overlap a fraction of a chunk (that's the whole
        // point of stitching), so at most pixels only 1 of N contributors
        // actually has real data there - stacking blindly across all N via
        // MedianStack (e.g. median of real,0,0) collapses those pixels to
        // 0, effectively erasing most of the chunk outside the few regions
        // where a majority of contributors happen to overlap. Combine
        // per-pixel instead: only contributors whose warped RGB triple
        // isn't exactly (0,0,0) - "no data here" - participate in that
        // pixel's result. A pixel with zero valid contributors stays 0
        // (genuinely nothing covers it); this approximates "all-zero ==
        // no data" rather than tracking a real per-pixel validity mask
        // through WarpPerspective, which would need touching every SIMD
        // tier and the CUDA kernel for a proper fix.
        void CombineIgnoringGaps(const std::vector<std::vector<unsigned short>>& warpedBuffers,
                                  size_t chunkPixelCount, std::vector<unsigned short>& outResult)
        {
            outResult.assign(chunkPixelCount, 0);

            std::vector<unsigned short> channelSamples;
            channelSamples.reserve(warpedBuffers.size());

            for (size_t idx = 0; idx < chunkPixelCount; idx += 3)
            {
                for (int c = 0; c < 3; ++c)
                {
                    channelSamples.clear();
                    for (const auto& buf : warpedBuffers)
                    {
                        bool hasData = buf[idx] != 0 || buf[idx + 1] != 0 || buf[idx + 2] != 0;
                        if (hasData) channelSamples.push_back(buf[idx + c]);
                    }
                    if (channelSamples.empty()) continue; // stays 0 - nothing covers this pixel

                    std::sort(channelSamples.begin(), channelSamples.end());
                    size_t mid = channelSamples.size() / 2;
                    outResult[idx + c] = (channelSamples.size() % 2 == 0)
                        ? static_cast<unsigned short>((channelSamples[mid - 1] + channelSamples[mid] + 1) / 2)
                        : channelSamples[mid];
                }
            }
        }
    }

    RenderExecutor::RenderExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                    std::shared_ptr<Compute::IComputeBackend> cudaPipeline)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_cudaPipeline(std::move(cudaPipeline))
    {
    }

    bool RenderExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;
        if (!m_cudaPipeline) return false;

        const ChunkModel* chunk = nullptr;
        for (const auto& c : m_projectManager.GetChunks())
        {
            if (c.id == task.unitKey)
            {
                chunk = &c;
                break;
            }
        }
        if (!chunk) return false;

        auto contributorIds = m_projectManager.GetChunkContributors(task.unitKey);
        if (contributorIds.empty()) return false;

        if (contributorIds.size() > kMaxContributorsForMedianStack)
        {
            // Documented limitation - MedianStack caps at 32 inputs.
            contributorIds.resize(kMaxContributorsForMedianStack);
        }

        size_t chunkPixelCount = static_cast<size_t>(chunk->width) * static_cast<size_t>(chunk->height) * 3;

        std::vector<std::vector<unsigned short>> warpedBuffers;
        warpedBuffers.reserve(contributorIds.size());

        for (int imageId : contributorIds)
        {
            const InputImageModel* image = nullptr;
            for (const auto& img : m_projectManager.GetInputImages())
            {
                if (img.id == imageId)
                {
                    image = &img;
                    break;
                }
            }
            if (!image) continue;

            auto colorBlobId = FindColorBlobId(m_projectManager, imageId);
            if (!colorBlobId.has_value()) continue;

            auto sourceBuffer = m_storageEngine.ReadPixelBuffer(colorBlobId.value());
            if (!sourceBuffer.has_value()) continue;

            // image->homography maps this image's own LOCAL pixel
            // coordinates into the shared world/canvas frame (Align/
            // BundleAdjustment's convention - confirmed via
            // WindowsApp.Tests/BundleAdjustmentTests.cpp), but
            // WarpPerspective's own `homography` parameter is documented
            // (WarpPerspectiveKernels.h) to be the OPPOSITE direction -
            // destination(world/chunk) -> source(local) - for its
            // backward-mapping sampler. Passing the forward homography
            // directly (as this code used to) is only harmless for the
            // reference image (identity is its own inverse); every other
            // image would sample from the wrong location and silently
            // warp to all-zero.
            float inverseHomography[9];
            if (!InvertHomography3x3(image->homography.h.data(), inverseHomography)) continue;

            std::vector<unsigned short> warped(chunkPixelCount, 0);
            Compute::ComputeResult warpResult = m_cudaPipeline->WarpPerspective(
                sourceBuffer->data.data(), sourceBuffer->width, sourceBuffer->height,
                warped.data(), chunk->width, chunk->height,
                inverseHomography, chunk->x_offset, chunk->y_offset);
            if (warpResult != Compute::ComputeResult::SUCCESS) continue;

            if (std::fabs(image->gain - 1.0f) > 1e-6f)
            {
                m_cudaPipeline->ApplyGain(warped.data(), static_cast<int>(chunkPixelCount), image->gain);
            }

            warpedBuffers.push_back(std::move(warped));
        }

        if (warpedBuffers.empty()) return false;

        PixelBuffer result;
        result.width = chunk->width;
        result.height = chunk->height;

        // Multi-band blending isn't wired in here (see this plan's own
        // header note: CudaPipeline::MultiBandBlend is pairwise, doesn't
        // fit N-way chunk contributors as-is) - a per-pixel, gap-aware
        // median (see CombineIgnoringGaps above) is the consensus/seam-
        // handling step for this v1 pass, not IComputeBackend::MedianStack
        // directly (which has no notion of per-pixel coverage gaps).
        CombineIgnoringGaps(warpedBuffers, chunkPixelCount, result.data);

        auto blobId = m_storageEngine.WritePixelBuffer(result, "rendered_chunk_rgb48");
        if (!blobId.has_value()) return false;

        task.outputBlobId = blobId;
        return true;
    }
}
