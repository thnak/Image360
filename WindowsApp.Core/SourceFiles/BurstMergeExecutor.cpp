#include "pch.h"
#include "HeaderFiles/BurstMergeExecutor.h"
#include "HeaderFiles/BurstCommon.h"
#include "HeaderFiles/BlockMatchAlignKernel.h"

#include <string>
#include <vector>

namespace WindowsApp::Core
{
    BurstMergeExecutor::BurstMergeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                            std::shared_ptr<Compute::IComputeBackend> computeBackend)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_computeBackend(std::move(computeBackend))
    {
    }

    bool BurstMergeExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;

        if (m_projectManager.GetBurstMode() != BurstMode::MFNR)
        {
            task.errorMessage = "BurstMergeExecutor: not yet implemented for this BurstMode (only MFNR is implemented).";
            return false;
        }

        switch (task.stage)
        {
        case PipelineStage::BURST_MERGE:  return ExecuteMerge(task);
        case PipelineStage::BURST_FINISH: return ExecuteFinish(task);
        default: return false;
        }
    }

    bool BurstMergeExecutor::ExecuteMerge(Task& task)
    {
        std::vector<Task> alignTasks = m_projectManager.GetTasksForStage(PipelineStage::BURST_ALIGN);
        if (alignTasks.empty())
        {
            task.errorMessage = "No BURST_ALIGN tasks found.";
            return false;
        }

        int referenceId = BurstReferenceImageId(m_projectManager.GetInputImages());

        const Task* referenceTask = nullptr;
        std::vector<const Task*> nonReferenceTasks;
        for (const auto& alignTask : alignTasks)
        {
            if (alignTask.status != TaskStatus::COMPLETED || !alignTask.outputBlobId.has_value())
            {
                task.errorMessage = "A BURST_ALIGN task has not completed yet.";
                return false;
            }

            int imageId = 0;
            try { imageId = std::stoi(alignTask.unitKey); }
            catch (...) { task.errorMessage = "Malformed BURST_ALIGN unitKey."; return false; }

            if (imageId == referenceId) referenceTask = &alignTask;
            else nonReferenceTasks.push_back(&alignTask);
        }
        if (!referenceTask)
        {
            task.errorMessage = "Reference frame's BURST_ALIGN task not found.";
            return false;
        }

        auto referenceBuffer = m_storageEngine.ReadPixelBuffer(*referenceTask->outputBlobId);
        if (!referenceBuffer.has_value())
        {
            task.errorMessage = "Failed to read the reference frame's decoded blob.";
            return false;
        }

        int expectedTilesX = (referenceBuffer->width + kBurstTileSize - 1) / kBurstTileSize;
        int expectedTilesY = (referenceBuffer->height + kBurstTileSize - 1) / kBurstTileSize;

        std::vector<PixelBuffer> nonReferenceBuffers;
        std::vector<std::vector<Compute::TileOffset>> perFrameOffsets;
        nonReferenceBuffers.reserve(nonReferenceTasks.size());
        perFrameOffsets.reserve(nonReferenceTasks.size());

        for (const Task* nonRefTask : nonReferenceTasks)
        {
            auto buffer = m_storageEngine.ReadPixelBuffer(*nonRefTask->outputBlobId);
            if (!buffer.has_value())
            {
                task.errorMessage = "Failed to read a burst frame's decoded blob.";
                return false;
            }
            if (buffer->width != referenceBuffer->width || buffer->height != referenceBuffer->height)
            {
                task.errorMessage = "Burst frame dimensions don't match the reference frame.";
                return false;
            }

            std::vector<Compute::TileOffset> offsets;
            int tilesX = 0, tilesY = 0;
            if (!Kernels::DeserializeTileOffsets(nonRefTask->checkpointJson, offsets, tilesX, tilesY)
                || tilesX != expectedTilesX || tilesY != expectedTilesY)
            {
                task.errorMessage = "Failed to deserialize (or inconsistent) TileOffset field for a burst frame.";
                return false;
            }

            nonReferenceBuffers.push_back(std::move(*buffer));
            perFrameOffsets.push_back(std::move(offsets));
        }

        std::vector<const unsigned short*> framePtrs;
        framePtrs.reserve(1 + nonReferenceBuffers.size());
        framePtrs.push_back(referenceBuffer->data.data());
        for (const auto& buffer : nonReferenceBuffers) framePtrs.push_back(buffer.data.data());

        std::vector<const Compute::TileOffset*> offsetPtrs;
        offsetPtrs.reserve(perFrameOffsets.size());
        for (const auto& offsets : perFrameOffsets) offsetPtrs.push_back(offsets.data());

        if (!m_computeBackend) return false;

        PixelBuffer merged;
        merged.width = referenceBuffer->width;
        merged.height = referenceBuffer->height;
        merged.data.resize(referenceBuffer->data.size());

        Compute::ComputeResult result = m_computeBackend->RobustMergeAccumulate(
            framePtrs.data(), static_cast<int>(framePtrs.size()), offsetPtrs.data(),
            merged.width, merged.height, kBurstTileSize, expectedTilesX, expectedTilesY,
            kBurstMergeSigma, merged.data.data());
        if (result != Compute::ComputeResult::SUCCESS)
        {
            task.errorMessage = m_computeBackend->GetLastError();
            return false;
        }

        auto blobId = m_storageEngine.WritePixelBuffer(merged, "raw_rgb48");
        if (!blobId.has_value())
        {
            task.errorMessage = "Failed to write the merged buffer.";
            return false;
        }

        task.outputBlobId = blobId;
        return true;
    }

    bool BurstMergeExecutor::ExecuteFinish(Task& task)
    {
        std::vector<Task> mergeTasks = m_projectManager.GetTasksForStage(PipelineStage::BURST_MERGE);
        if (mergeTasks.size() != 1 || mergeTasks[0].status != TaskStatus::COMPLETED
            || !mergeTasks[0].outputBlobId.has_value())
        {
            task.errorMessage = "BURST_MERGE task has not completed yet.";
            return false;
        }

        auto bytes = m_storageEngine.ReadBlob(*mergeTasks[0].outputBlobId);
        if (!bytes.has_value())
        {
            task.errorMessage = "Failed to read BURST_MERGE's output blob.";
            return false;
        }

        // Passthrough: copies the exact bytes (including WritePixelBuffer's
        // embedded width/height header) forward under a new blob, rather
        // than round-tripping through PixelBuffer - MFNR's BURST_FINISH
        // has no actual transformation to apply (see this class's header
        // comment).
        auto blobId = m_storageEngine.WriteBlob(bytes->data(), bytes->size(), "raw_rgb48");
        if (!blobId.has_value())
        {
            task.errorMessage = "Failed to write BURST_FINISH's output blob.";
            return false;
        }

        task.outputBlobId = blobId;
        return true;
    }
}
