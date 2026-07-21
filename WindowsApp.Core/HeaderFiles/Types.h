#pragma once
#include <string>
#include <vector>
#include <array>
#include <optional>
#include <cstdint>
#include <stop_token>

namespace WindowsApp::Core
{
    enum class ChunkStatus
    {
        PENDING,
        PROCESSING,
        COMPLETED,
        FAILED
    };

    // Routes RawIngest to the GPU demosaic path (BAYER) or LibRaw's own
    // CPU dcraw_process() (X_TRANS, FOVEON - a deliberate, documented
    // exception per docs/ARCHITECTURE.md SS4.1, not a gap to close).
    // STANDARD_RGB is not a CFA at all - it routes RawIngest around
    // ImageLoader/LibRaw entirely to JpegCodec (.jpg/.jpeg) or vendored
    // stb_image (.png/.bmp/.gif/.tga/.tif/.tiff), since any standard
    // consumer image format (what most phone cameras produce) is already
    // demosaiced RGB with no sensor mosaic to reconstruct. See
    // ImageLoader::IsJpegFile / IsStandardImageFile.
    enum class CfaType
    {
        BAYER,
        X_TRANS,
        FOVEON,
        STANDARD_RGB,
        UNKNOWN
    };

    enum class PipelineStage
    {
        IDLE,
        STAGE0_INGEST,
        STAGE1_ALIGN,
        STAGE2_OPTIMIZE,
        STAGE3_RENDER,
        // Burst-mode pipeline family (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS3,
        // SS8 Phase 0) - align + merge + finish, driven by PipelineDriver
        // when ProjectManager::GetProjectType() == ProjectType::BURST
        // instead of the four STAGE*/panorama stages above. Which merge
        // algorithm BURST_MERGE actually runs (robust weighted merge for
        // MFNR, FFT/Wiener-shrinkage for HDR+, structure-tensor kernel
        // regression for Night Sight/Super Res Zoom) is BurstMode-selected
        // executor logic, not a separate stage - see
        // docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.3's correction.
        BURST_ALIGN,
        BURST_MERGE,
        BURST_FINISH,
        COMPLETED,
        CANCELLED,
        FAILED
    };

    // A project's pipeline family - selects which PipelineStage sequence
    // PipelineDriver::Run drives. Every project created before this existed
    // has no 'project_type' key in its project_metadata table and loads as
    // PANORAMA (ProjectManager's constructor default), so no migration is
    // needed for existing .vfp files.
    enum class ProjectType
    {
        PANORAMA,
        BURST
    };

    // Which burst algorithm BURST_MERGE's registered executor should run.
    // NONE is what a PANORAMA project reports (not a 5th "no mode" burst
    // mode) - kept as a separate enum from ProjectType rather than folding
    // both into one 5-way enum, so callers never have to special-case which
    // values of one enum "actually mean panorama."
    enum class BurstMode
    {
        NONE,
        MFNR,
        HDR_PLUS,
        NIGHT_SIGHT,
        SUPER_RES
    };

    enum class TaskStatus
    {
        PENDING,
        RUNNING,
        COMPLETED,
        FAILED,
        CANCELLED
    };

    struct Task
    {
        int64_t taskId = 0;
        PipelineStage stage = PipelineStage::IDLE;
        std::string unitKind;   // "image" | "image_band" | "pair" | "chunk" | "ba_checkpoint"
        std::string unitKey;    // e.g. "img_7", "img_7:band_3", "img_2:img_9", "C_4_2"
        TaskStatus status = TaskStatus::PENDING;
        int attemptCount = 0;
        std::optional<int64_t> outputBlobId;
        std::string checkpointJson;
        // Transient - never persisted to the DB (GetTasksForStage() et al.
        // leave it default-empty). An executor's Execute(Task&, ...) can
        // set this before returning false so the scheduler's failure
        // callback/log has something more specific than "returned false".
        std::string errorMessage;
    };

    struct BlobDirectoryEntry
    {
        int64_t blobId = 0;
        std::wstring shardFile;
        int64_t offset = 0;
        int64_t length = 0;
        std::optional<int64_t> compressedLength;
        std::string formatTag; // e.g. "raw_rgb48", "gdeflate", "nvjpeg"
    };

    // Placed here (not a new header) because it's used by Task-adjacent
    // signatures across both WindowsApp.Core and, later, WindowsApp.Compute.
    using CancellationToken = std::stop_token;

    struct PixelBuffer
    {
        int width = 0;
        int height = 0;
        std::vector<unsigned short> data; // 16-bit channel representation (RGB48 or Grayscale)
    };

    struct Homography
    {
        std::array<float, 9> h = { 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    };

    struct ChunkModel
    {
        std::string id;          // Format: "C_X_Y"
        int x_offset = 0;        // X coordinate of top-left corner on the virtual grid
        int y_offset = 0;        // Y coordinate of top-left corner on the virtual grid
        int width = 4096;
        int height = 4096;
        ChunkStatus status = ChunkStatus::PENDING;
        std::wstring cache_path; // Win32 Memory-Mapped File path for cache
    };
}
