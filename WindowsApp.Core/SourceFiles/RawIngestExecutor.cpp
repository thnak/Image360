#include "pch.h"
#include "HeaderFiles/RawIngestExecutor.h"
#include "HeaderFiles/ImageLoader.h"
#include "HeaderFiles/PlatformFile.h"
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    RawIngestExecutor::RawIngestExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                          std::shared_ptr<Compute::IComputeBackend> cudaPipeline,
                                          std::shared_ptr<Compute::IImageCodec> jpegCodec)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_cudaPipeline(std::move(cudaPipeline))
        , m_jpegCodec(std::move(jpegCodec))
    {
    }

    namespace
    {
        bool DecodeJpegImage(const InputImageModel& image, std::shared_ptr<Compute::IImageCodec> jpegCodec,
                              PixelBuffer& outBuffer)
        {
            if (!jpegCodec) return false;

            PlatformFile file;
            if (!file.Open(image.file_path, FileOpenMode::ReadOnly)) return false;

            std::vector<unsigned char> jpegBytes(static_cast<size_t>(file.Size()));
            if (!jpegBytes.empty() && !file.Read(jpegBytes.data(), jpegBytes.size())) return false;

            unsigned char* rgb8 = nullptr;
            int width = 0, height = 0;
            Compute::ComputeResult result = jpegCodec->Decode(jpegBytes.data(), jpegBytes.size(), &rgb8, &width, &height);
            if (result != Compute::ComputeResult::SUCCESS || !rgb8) return false;

            // Widen 8-bit -> 16-bit (RawIngest's contract, and every later
            // stage's input, is RGB48 regardless of source format) - *257 is
            // the exact linear scaling (255*257 == 65535), not a lossy
            // approximation like a left-shift would be for non-0xFF inputs.
            outBuffer.width = width;
            outBuffer.height = height;
            outBuffer.data.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 3);
            for (size_t i = 0; i < outBuffer.data.size(); ++i)
            {
                outBuffer.data[i] = static_cast<unsigned short>(rgb8[i]) * 257;
            }

            jpegCodec->FreeDecoded(rgb8);
            return true;
        }
    }

    bool DecodeInputImage(const InputImageModel& image,
                           std::shared_ptr<Compute::IComputeBackend> computeBackend,
                           std::shared_ptr<Compute::IImageCodec> jpegCodec,
                           PixelBuffer& outBuffer)
    {
        // Standard consumer image formats (CfaType::STANDARD_RGB - see
        // AddInputImage's caller) never touch ImageLoader/LibRaw at all:
        // they're not a RAW file and have no CFA to demosaic, so LibRaw's
        // Open() would just fail on them (or worse, misbehave on a format
        // it wasn't designed for). JPEG goes through JpegCodec/NvJpegCodec
        // (GPU-capable where available); everything else (PNG/BMP/GIF/
        // TGA/TIFF) through vendored stb_image.
        if (image.cfaType == CfaType::STANDARD_RGB)
        {
            return IsJpegFile(image.file_path)
                ? DecodeJpegImage(image, std::move(jpegCodec), outBuffer)
                : DecodeStandardImage(image.file_path, outBuffer);
        }

        ImageLoader loader;
        if (!loader.Open(image.file_path)) return false;

        RawPlane plane;
        if (!loader.UnpackRaw(plane)) return false;

        switch (plane.cfaType)
        {
        case CfaType::BAYER:
        {
            if (!computeBackend) return false;

            outBuffer.width = plane.width;
            outBuffer.height = plane.height;
            outBuffer.data.resize(static_cast<size_t>(plane.width) * static_cast<size_t>(plane.height) * 3);

            Compute::ComputeResult result = computeBackend->DemosaicBayer(
                plane.cfaData.data(), plane.width, plane.height,
                static_cast<unsigned short>(plane.blackLevel), plane.camMul, plane.rgbCam,
                plane.filters, outBuffer.data.data());
            return result == Compute::ComputeResult::SUCCESS;
        }
        case CfaType::X_TRANS:
        case CfaType::FOVEON:
            // Deliberate, documented exception per docs/ARCHITECTURE.md
            // SS4.1 - exotic CFAs fall back to LibRaw's own CPU
            // dcraw_process() for demosaic, not new GPU code.
            return loader.DecodeFull(outBuffer);
        case CfaType::UNKNOWN:
        default:
            return false;
        }
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

        PixelBuffer buffer;
        if (!DecodeInputImage(*image, m_cudaPipeline, m_jpegCodec, buffer)) return false;

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
