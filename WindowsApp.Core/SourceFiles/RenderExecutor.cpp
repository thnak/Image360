#include "pch.h"
#include "HeaderFiles/RenderExecutor.h"
#include <cmath>
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        constexpr size_t kMaxContributorsForMedianStack = 32; // CudaPipeline::MedianStack's hard cap

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
                                    std::shared_ptr<Compute::CudaPipeline> cudaPipeline)
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

            std::vector<unsigned short> warped(chunkPixelCount, 0);
            Compute::ComputeResult warpResult = m_cudaPipeline->WarpPerspective(
                sourceBuffer->data.data(), sourceBuffer->width, sourceBuffer->height,
                warped.data(), chunk->width, chunk->height,
                image->homography.h.data(), chunk->x_offset, chunk->y_offset);
            if (warpResult != Compute::ComputeResult::SUCCESS) continue;

            if (std::fabs(image->gain - 1.0f) > 1e-6f)
            {
                m_cudaPipeline->ApplyGain(warped.data(), static_cast<int>(chunkPixelCount), image->gain);
            }

            warpedBuffers.push_back(std::move(warped));
        }

        if (warpedBuffers.empty()) return false;

        std::vector<const unsigned short*> inputPtrs;
        inputPtrs.reserve(warpedBuffers.size());
        for (const auto& buf : warpedBuffers)
        {
            inputPtrs.push_back(buf.data());
        }

        PixelBuffer result;
        result.width = chunk->width;
        result.height = chunk->height;
        result.data.resize(chunkPixelCount);

        // Multi-band blending isn't wired in here (see this plan's own
        // header note: CudaPipeline::MultiBandBlend is pairwise, doesn't
        // fit N-way chunk contributors as-is) - MedianStack alone is the
        // consensus/seam-handling step for this v1 pass.
        Compute::ComputeResult stackResult = m_cudaPipeline->MedianStack(
            inputPtrs.data(), static_cast<int>(inputPtrs.size()),
            result.data.data(), chunk->width, chunk->height);
        if (stackResult != Compute::ComputeResult::SUCCESS) return false;

        auto blobId = m_storageEngine.WritePixelBuffer(result, "rendered_chunk_rgb48");
        if (!blobId.has_value()) return false;

        task.outputBlobId = blobId;
        return true;
    }
}
