#pragma once
#include "Types.h"
#include <string>
#include <memory>

namespace WindowsApp::Core
{
    struct ImageMetadata
    {
        int width = 0;
        int height = 0;
        int bitsPerSample = 16;
        int colors = 3;
        std::wstring cameraMake;
        std::wstring cameraModel;
        float isoSpeed = 0.0f;
        float shutterSpeed = 0.0f;
        float aperture = 0.0f;
        float focalLength = 0.0f;
        int orientation = 1; // EXIF orientation
    };

    class ImageLoader
    {
    public:
        ImageLoader();
        ~ImageLoader();

        // Disable copy
        ImageLoader(const ImageLoader&) = delete;
        ImageLoader& operator=(const ImageLoader&) = delete;

        // Open a RAW file (RAF, CR2, NEF, ARW, DNG, etc.)
        bool Open(const std::wstring& filePath);
        void Close();
        bool IsOpen() const;

        // Get image metadata without decoding
        bool GetMetadata(ImageMetadata& metadata) const;

        // Decode full resolution to PixelBuffer (16-bit RGB48)
        bool DecodeFull(PixelBuffer& output);

        // Decode downsampled (scalePercent: 1-100)
        bool DecodeThumbnail(PixelBuffer& output, int scalePercent = 10);

        // Decode a specific ROI (for Stage 3 tile processing)
        // x, y, width, height in pixel coordinates of the full image
        bool DecodeROI(int x, int y, int width, int height, PixelBuffer& output);

        // Get the last error message
        std::wstring GetLastError() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        bool DecodeInternal(PixelBuffer& output, int scalePercent, int roiX, int roiY, int roiW, int roiH);
    };
}
