#include "pch.h"
#include "HeaderFiles/BurstAlignExecutor.h"
#include "HeaderFiles/RawIngestExecutor.h"
#include "HeaderFiles/BlockMatchAlignKernel.h"
#include "HeaderFiles/BurstCommon.h"

#include <algorithm>
#include <string>

namespace WindowsApp::Core
{
    namespace
    {
        const InputImageModel* FindImage(const std::vector<InputImageModel>& images, int id)
        {
            for (const auto& img : images)
                if (img.id == id) return &img;
            return nullptr;
        }
    }

    BurstAlignExecutor::BurstAlignExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                            std::shared_ptr<Compute::IComputeBackend> computeBackend,
                                            std::shared_ptr<Compute::IImageCodec> jpegCodec)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_computeBackend(std::move(computeBackend))
        , m_jpegCodec(std::move(jpegCodec))
    {
    }

    bool BurstAlignExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;

        int imageId = 0;
        try { imageId = std::stoi(task.unitKey); }
        catch (...) { return false; }

        const auto& images = m_projectManager.GetInputImages();
        const InputImageModel* image = FindImage(images, imageId);
        if (!image) return false;

        PixelBuffer buffer;
        if (!DecodeInputImage(*image, m_computeBackend, m_jpegCodec, buffer)) return false;

        auto blobId = m_storageEngine.WritePixelBuffer(buffer, "raw_rgb48");
        if (!blobId.has_value()) return false;
        task.outputBlobId = blobId;

        int referenceId = BurstReferenceImageId(images);
        if (imageId == referenceId)
        {
            // The reference frame doesn't align against itself.
            return true;
        }

        const InputImageModel* referenceImage = FindImage(images, referenceId);
        if (!referenceImage) return false;

        PixelBuffer referenceBuffer;
        if (!DecodeInputImage(*referenceImage, m_computeBackend, m_jpegCodec, referenceBuffer)) return false;

        if (referenceBuffer.width != buffer.width || referenceBuffer.height != buffer.height)
        {
            task.errorMessage = "Burst frame dimensions don't match the reference frame.";
            return false;
        }

        if (!m_computeBackend) return false;

        int tilesX = (buffer.width + kBurstTileSize - 1) / kBurstTileSize;
        int tilesY = (buffer.height + kBurstTileSize - 1) / kBurstTileSize;
        std::vector<Compute::TileOffset> offsets(static_cast<size_t>(tilesX) * tilesY);

        Compute::ComputeResult result = m_computeBackend->BlockMatchAlign(
            referenceBuffer.data.data(), buffer.data.data(),
            buffer.width, buffer.height, kBurstTileSize, kBurstSearchRadius,
            offsets.data(), tilesX, tilesY);
        if (result != Compute::ComputeResult::SUCCESS)
        {
            task.errorMessage = m_computeBackend->GetLastError();
            return false;
        }

        task.checkpointJson = Kernels::SerializeTileOffsets(offsets, tilesX, tilesY);
        // Setting task.checkpointJson alone is NOT persisted by
        // TaskScheduler (it only commits status/outputBlobId once Execute
        // returns) - must be written explicitly, matching
        // OptimizeExecutor's existing UpdateTaskCheckpoint calls for its
        // own (differently-purposed) checkpointJson use.
        m_projectManager.UpdateTaskCheckpoint(task.taskId, task.checkpointJson);
        return true;
    }
}
