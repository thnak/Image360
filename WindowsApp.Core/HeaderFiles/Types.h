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
    enum class CfaType
    {
        BAYER,
        X_TRANS,
        FOVEON,
        UNKNOWN
    };

    enum class PipelineStage
    {
        IDLE,
        STAGE0_INGEST,
        STAGE1_ALIGN,
        STAGE2_OPTIMIZE,
        STAGE3_RENDER,
        COMPLETED,
        CANCELLED,
        FAILED
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
