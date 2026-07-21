#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "IComputeBackend.h"
#include "IImageCodec.h"
#include <memory>

namespace WindowsApp::Core
{
    // Decodes one InputImageModel to a PixelBuffer (RGB48), dispatching by
    // CfaType: STANDARD_RGB -> JpegCodec/stb_image, BAYER -> GPU demosaic
    // via computeBackend, X_TRANS/FOVEON -> LibRaw's own CPU
    // dcraw_process(). Shared by RawIngestExecutor (panorama's
    // STAGE0_INGEST) and BurstAlignExecutor (burst mode's BURST_ALIGN) -
    // extracted here rather than duplicated, since both need exactly this
    // logic and nothing executor-specific. Free function, not a method:
    // needs no executor state, just the inputs.
    bool DecodeInputImage(const InputImageModel& image,
                           std::shared_ptr<Compute::IComputeBackend> computeBackend,
                           std::shared_ptr<Compute::IImageCodec> jpegCodec,
                           PixelBuffer& outBuffer);

    // First real (non-stub) ITaskExecutor - docs/ARCHITECTURE.md SS4.1.
    // unit_kind = "image", unit_key = the input image's DB id.
    class RawIngestExecutor : public ITaskExecutor
    {
    public:
        // jpegCodec decodes CfaType::JPEG_RGB inputs (plain consumer
        // JPEGs, e.g. straight off a phone camera - no CFA to demosaic);
        // the RAW formats (BAYER/X_TRANS/FOVEON) never touch it.
        RawIngestExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                           std::shared_ptr<Compute::IComputeBackend> cudaPipeline,
                           std::shared_ptr<Compute::IImageCodec> jpegCodec);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::IComputeBackend> m_cudaPipeline;
        std::shared_ptr<Compute::IImageCodec> m_jpegCodec;
    };
}
