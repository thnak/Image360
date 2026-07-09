#pragma once
#include "Types.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "IImageCodec.h"
#include <memory>
#include <string>

namespace WindowsApp::Core
{
    struct ChunkPlacement
    {
        int destX = 0;
        int destY = 0;
        int destW = 0;
        int destH = 0;
    };

    // Pure arithmetic, no I/O - testable without a GPU.
    float ComputeExportScale(int totalWidth, int totalHeight, int maxDimension);
    ChunkPlacement ComputeChunkPlacement(const ChunkModel& chunk, float scale);

    // Not a pipeline stage/Task - a user-triggered, on-demand action that
    // runs once a project's STAGE3_RENDER tasks are all COMPLETED. Reads
    // their output blobs directly and writes a plain .jpg file to a
    // user-chosen path outside the project container (not through
    // StorageEngine/blob_directory - this is an exported artifact, not
    // internal pipeline state).
    class PanoramaExporter
    {
    public:
        PanoramaExporter(ProjectManager& projectManager, StorageEngine& storageEngine,
                          std::shared_ptr<Compute::IImageCodec> nvJpegCodec);

        // maxDimension: the exported preview's longest edge in pixels - a
        // multi-gigapixel archival render is never exported at full
        // resolution as a JPEG preview, the whole point of this being a
        // separate, smaller "preview/share" path (docs/ARCHITECTURE.md SS4.5).
        // Returns false (expected failure) rather than throwing.
        bool ExportPreviewJpeg(const std::wstring& destPath, int maxDimension, int quality = 90);

    private:
        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::IImageCodec> m_nvJpegCodec;
    };
}
