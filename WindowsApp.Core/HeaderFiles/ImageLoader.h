#pragma once
#include "Types.h"
#include <string>
#include <vector>
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

    // Result of unpack()-only access (no dcraw_process()) - the raw
    // sensor plane plus just enough metadata to demosaic it on the GPU
    // (docs/ARCHITECTURE.md SS4.1).
    struct RawPlane
    {
        int width = 0;
        int height = 0;
        std::vector<unsigned short> cfaData; // one sample per pixel, raw sensor values
        CfaType cfaType = CfaType::UNKNOWN;
        uint32_t filters = 0;   // LibRaw's CFA pattern encoding - needed by
                                 // the GPU demosaic kernels' per-pixel
                                 // color-channel lookup, not just cfaType
                                 // classification
        unsigned blackLevel = 0;
        float camMul[4] = { 1.0f, 1.0f, 1.0f, 1.0f };  // per-channel WB multipliers
        float rgbCam[3][4] = {};                        // camera RGB -> sRGB matrix
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

        // Stops after LibRaw's unpack() (already done as part of Open())
        // - no dcraw_process(). Must be called on an already-Open()'d
        // file. Populates RawPlane::cfaType so callers can route
        // Bayer-vs-exotic without duplicating that detection logic.
        bool UnpackRaw(RawPlane& output);

        // Extracts the embedded preview JPEG bytes (EXIF IFD1 / LibRaw
        // thumbnail) without touching the full-res CFA plane. Returns
        // false (not a JPEG format thumbnail, or none present) for
        // non-JPEG embedded thumbnails - an expected, documented gap,
        // not silently mishandled.
        bool GetEmbeddedPreviewJpeg(std::vector<unsigned char>& jpegBytes);

        // Get the last error message
        std::wstring GetLastError() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;

        bool DecodeInternal(PixelBuffer& output, int scalePercent, int roiX, int roiY, int roiW, int roiH);
    };
}
