#pragma once
#include <string>
#include <vector>
#include <array>

namespace WindowsApp::Core
{
    enum class ChunkStatus
    {
        PENDING,
        PROCESSING,
        COMPLETED,
        FAILED
    };

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
