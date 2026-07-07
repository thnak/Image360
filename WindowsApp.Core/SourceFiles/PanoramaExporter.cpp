#include "pch.h"
#include "HeaderFiles/PanoramaExporter.h"
#include <algorithm>
#include <vector>

namespace WindowsApp::Core
{
    float ComputeExportScale(int totalWidth, int totalHeight, int maxDimension)
    {
        int longestEdge = (std::max)(totalWidth, totalHeight);
        if (longestEdge <= 0) return 1.0f;
        return static_cast<float>(maxDimension) / static_cast<float>(longestEdge);
    }

    ChunkPlacement ComputeChunkPlacement(const ChunkModel& chunk, float scale)
    {
        ChunkPlacement placement;
        placement.destX = static_cast<int>(chunk.x_offset * scale);
        placement.destY = static_cast<int>(chunk.y_offset * scale);
        placement.destW = (std::max)(1, static_cast<int>(chunk.width * scale));
        placement.destH = (std::max)(1, static_cast<int>(chunk.height * scale));
        return placement;
    }

    PanoramaExporter::PanoramaExporter(ProjectManager& projectManager, StorageEngine& storageEngine,
                                        std::shared_ptr<Compute::NvJpegCodec> nvJpegCodec)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_nvJpegCodec(std::move(nvJpegCodec))
    {
    }

    bool PanoramaExporter::ExportPreviewJpeg(const std::wstring& destPath, int maxDimension, int quality)
    {
        if (!m_nvJpegCodec) return false;
        if (maxDimension <= 0) return false;

        auto renderTasks = m_projectManager.GetTasksForStage(PipelineStage::STAGE3_RENDER);
        if (renderTasks.empty()) return false;
        for (const auto& task : renderTasks)
        {
            if (task.status != TaskStatus::COMPLETED) return false;
        }

        int totalWidth = m_projectManager.GetTotalWidth();
        int totalHeight = m_projectManager.GetTotalHeight();
        if (totalWidth <= 0 || totalHeight <= 0) return false;

        float scale = ComputeExportScale(totalWidth, totalHeight, maxDimension);
        int outWidth = (std::max)(1, static_cast<int>(totalWidth * scale));
        int outHeight = (std::max)(1, static_cast<int>(totalHeight * scale));

        std::vector<unsigned char> composite(static_cast<size_t>(outWidth) * static_cast<size_t>(outHeight) * 3, 0);

        for (const auto& chunk : m_projectManager.GetChunks())
        {
            int64_t blobId = 0;
            bool found = false;
            for (const auto& task : renderTasks)
            {
                if (task.unitKind == "chunk" && task.unitKey == chunk.id && task.outputBlobId.has_value())
                {
                    blobId = task.outputBlobId.value();
                    found = true;
                    break;
                }
            }
            if (!found) continue; // an empty (no-contributor) chunk has no render task/output

            auto pixelBuffer = m_storageEngine.ReadPixelBuffer(blobId);
            if (!pixelBuffer.has_value()) continue;

            ChunkPlacement placement = ComputeChunkPlacement(chunk, scale);

            for (int dy = 0; dy < placement.destH; ++dy)
            {
                int outY = placement.destY + dy;
                if (outY < 0 || outY >= outHeight) continue;

                for (int dx = 0; dx < placement.destW; ++dx)
                {
                    int outX = placement.destX + dx;
                    if (outX < 0 || outX >= outWidth) continue;

                    int srcX = static_cast<int>(dx / scale);
                    int srcY = static_cast<int>(dy / scale);
                    if (srcX < 0 || srcX >= pixelBuffer->width || srcY < 0 || srcY >= pixelBuffer->height) continue;

                    int srcIdx = (srcY * pixelBuffer->width + srcX) * 3;
                    int dstIdx = (outY * outWidth + outX) * 3;

                    // 16-bit -> 8-bit via a straight bit shift, matching
                    // existing RGB48->display conversions elsewhere in
                    // this codebase - acceptable for a preview export,
                    // not a real tone-mapping curve.
                    composite[dstIdx]     = static_cast<unsigned char>(pixelBuffer->data[srcIdx] >> 8);
                    composite[dstIdx + 1] = static_cast<unsigned char>(pixelBuffer->data[srcIdx + 1] >> 8);
                    composite[dstIdx + 2] = static_cast<unsigned char>(pixelBuffer->data[srcIdx + 2] >> 8);
                }
            }
        }

        std::vector<unsigned char> jpegBytes;
        if (m_nvJpegCodec->Encode(composite.data(), outWidth, outHeight, quality, jpegBytes) != Compute::ComputeResult::SUCCESS)
        {
            return false;
        }

        HANDLE fileHandle = CreateFileW(
            destPath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) return false;

        DWORD written = 0;
        bool writeOk = WriteFile(fileHandle, jpegBytes.data(), static_cast<DWORD>(jpegBytes.size()), &written, nullptr)
            && written == jpegBytes.size();
        CloseHandle(fileHandle);

        return writeOk;
    }
}
