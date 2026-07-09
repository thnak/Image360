#include "pch.h"
#include "HeaderFiles/RawIngestExecutor.h"
#include "HeaderFiles/ImageLoader.h"
#include <string>

namespace WindowsApp::Core
{
    RawIngestExecutor::RawIngestExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                          std::shared_ptr<Compute::IComputeBackend> cudaPipeline)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_cudaPipeline(std::move(cudaPipeline))
    {
    }

    bool RawIngestExecutor::Execute(Task& task, CancellationToken token)
    {
        // Checked once at entry only - a dispatched unit of GPU work
        // always finishes once started (docs/ARCHITECTURE.md SS7.2).
        if (token.stop_requested()) return false;

        int imageId = 0;
        try
        {
            imageId = std::stoi(task.unitKey);
        }
        catch (...)
        {
            return false;
        }

        const InputImageModel* image = nullptr;
        for (const auto& img : m_projectManager.GetInputImages())
        {
            if (img.id == imageId)
            {
                image = &img;
                break;
            }
        }
        if (!image) return false;

        ImageLoader loader;
        if (!loader.Open(image->file_path)) return false;

        RawPlane plane;
        if (!loader.UnpackRaw(plane)) return false;

        PixelBuffer buffer;

        switch (plane.cfaType)
        {
        case CfaType::BAYER:
        {
            if (!m_cudaPipeline) return false;

            buffer.width = plane.width;
            buffer.height = plane.height;
            buffer.data.resize(static_cast<size_t>(plane.width) * static_cast<size_t>(plane.height) * 3);

            Compute::ComputeResult result = m_cudaPipeline->DemosaicBayer(
                plane.cfaData.data(), plane.width, plane.height,
                static_cast<unsigned short>(plane.blackLevel), plane.camMul, plane.rgbCam,
                plane.filters, buffer.data.data());
            if (result != Compute::ComputeResult::SUCCESS) return false;
            break;
        }
        case CfaType::X_TRANS:
        case CfaType::FOVEON:
            // Deliberate, documented exception per docs/ARCHITECTURE.md
            // SS4.1 - exotic CFAs fall back to LibRaw's own CPU
            // dcraw_process() for demosaic, not new GPU code.
            if (!loader.DecodeFull(buffer)) return false;
            break;
        case CfaType::UNKNOWN:
        default:
            return false;
        }

        auto blobId = m_storageEngine.WritePixelBuffer(buffer, "raw_rgb48");
        if (!blobId.has_value()) return false;

        // TaskScheduler's caller commits via ProjectManager::CommitTaskOutput
        // once Execute returns true - this class never calls it directly,
        // matching the durability-ordering contract from
        // docs/superpowers/plans/2026-07-07-storage-engine.md.
        task.outputBlobId = blobId;
        return true;
    }
}
