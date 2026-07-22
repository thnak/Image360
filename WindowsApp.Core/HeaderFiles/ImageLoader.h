#pragma once
#include "Types.h"
#include "IImageCodec.h"
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

    // True for a standard .jpg/.jpeg path (case-insensitive) - these
    // aren't RAW files (no CFA data to demosaic), so RawIngestExecutor
    // checks this before ever calling ImageLoader::Open, which would
    // otherwise hand a plain consumer JPEG to LibRaw and fail. Decoded via
    // JpegCodec/NvJpegCodec (GPU-capable where available), not stb_image.
    bool IsJpegFile(const std::wstring& path);

    // True for any other standard (non-RAW, non-JPEG) image format this
    // app can decode via vendored stb_image: PNG, BMP, GIF, TGA, TIFF.
    bool IsStandardImageFile(const std::wstring& path);

    // Header-only dimension read (no full decode) for any format
    // IsJpegFile or IsStandardImageFile recognizes - stb_image's info
    // reader covers JPEG too, so this is the one dimension-probe both
    // branches of RawIngestExecutor's non-RAW path share (see
    // MainWindow::RunStartButton_Click's picking loop).
    bool GetStandardImageDimensions(const std::wstring& path, int& outWidth, int& outHeight);

    // Decodes a non-RAW, non-JPEG standard image format (PNG/BMP/GIF/TGA/
    // TIFF, via vendored stb_image) to RGB48. JPEG specifically still
    // goes through JpegCodec/NvJpegCodec instead (see
    // RawIngestExecutor::ExecuteJpegIngest) - already GPU-capable where
    // available, no reason to route it through stb_image too.
    bool DecodeStandardImage(const std::wstring& path, PixelBuffer& output);

    // Same as DecodeStandardImage but leaves the result at 8 bits/channel
    // (no RGB48 widening) - for callers like DecodePreviewRgb8 that want
    // interleaved RGB8 directly, matching what IImageCodec::Decode
    // already produces for the JPEG case.
    bool DecodeStandardImageRgb8(const std::wstring& path, std::vector<unsigned char>& outRgb,
                                  int& outWidth, int& outHeight);

    // Decodes a small RGB8 preview of an input image for feature
    // detection (AlignExecutor) / gain-color-transfer (OptimizeExecutor)
    // - both need "a reasonably-sized RGB8 decode", not the full-res
    // RGB48 pipeline data. RAW files (cfaType != STANDARD_RGB) use their
    // embedded JPEG preview (cheap, no full demosaic, exactly what both
    // callers did before this existed); STANDARD_RGB files (plain JPEG/
    // PNG/BMP/GIF/TGA/TIFF - see RawIngestExecutor) have no separate
    // "preview" to extract, so this just decodes them directly (JPEG via
    // jpegCodec, everything else via stb_image). On failure, returns
    // false and outError describes what went wrong (mirrors the
    // task.errorMessage convention both callers already use).
    bool DecodePreviewRgb8(const std::wstring& filePath, CfaType cfaType, Compute::IImageCodec& jpegCodec,
                           std::vector<unsigned char>& outRgb, int& outWidth, int& outHeight,
                           std::string& outError);

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
