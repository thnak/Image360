#include "pch.h"
#include "HeaderFiles/RenderExecutor.h"
#include "HeaderFiles/HomographyMath.h"
#include "HeaderFiles/SeamBlendKernels.h"
#include <cmath>
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        // A generous sanity cap on contributors per chunk - Kernels::SeamBlend
        // has no hard cap of its own, this is just a defensive bound.
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

        // Seam-aware N-way multi-band blend (Kernels::SeamBlend) replaces
        // a plain per-pixel median/average across contributors: averaging
        // wherever contributors disagree (e.g. a nearby object at a
        // slightly different position in each warp due to real camera
        // parallax) produced ghosting - a translucent double-image of
        // that object. SeamBlend instead picks a single owning contributor
        // per pixel (steering the ownership boundary away from
        // high-disagreement content where possible) and blends only the
        // transition itself via a Laplacian pyramid, so there's no visible
        // hard seam either. Runs on the plain host RGB48 buffers already
        // produced above, independent of which IComputeBackend did the
        // warping.
        std::vector<const unsigned short*> warpedBufferPtrs;
        warpedBufferPtrs.reserve(warpedBuffers.size());
        for (const auto& buf : warpedBuffers) warpedBufferPtrs.push_back(buf.data());
        Kernels::SeamBlend::BlendChunkContributors(warpedBufferPtrs, chunk->width, chunk->height, result.data);

        auto blobId = m_storageEngine.WritePixelBuffer(result, "rendered_chunk_rgb48");
        if (!blobId.has_value()) return false;

        task.outputBlobId = blobId;
        return true;
    }
}
