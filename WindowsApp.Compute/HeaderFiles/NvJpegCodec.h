#pragma once

#ifdef WINDOWSAPPCOMPUTE_EXPORTS
#define COMPUTE_API __declspec(dllexport)
#else
#define COMPUTE_API __declspec(dllimport)
#endif

#include "CudaPipeline.h"
#include <cstddef>
#include <vector>

namespace WindowsApp { namespace Compute
{
    // GPU JPEG decode (docs/ARCHITECTURE.md SS4.5, preview decode).
    // Encode (export path) is a separate, later plan
    // (docs/superpowers/plans/2026-07-07-nvjpeg-export.md).
    //
    // Uses the CUDA Toolkit's nvjpeg.h (bundled, no separate SDK) - not
    // vendored in this repo the way LibRaw is, so exact symbol names here
    // are against the standard, stable nvJPEG C API and should be
    // spot-checked against the installed Toolkit's headers on first
    // Windows build.
    class COMPUTE_API NvJpegCodec
    {
    public:
        NvJpegCodec();
        ~NvJpegCodec();
        NvJpegCodec(const NvJpegCodec&) = delete;
        NvJpegCodec& operator=(const NvJpegCodec&) = delete;

        // Creates an nvjpegHandle_t against the current CUDA context -
        // call after CudaPipeline::Initialize() has selected a device.
        ComputeResult Initialize();
        void Shutdown();

        // Decodes a JPEG byte buffer to interleaved RGB8, allocating
        // outRgb (caller frees via FreeDecoded) - matches CudaPipeline's
        // existing malloc-per-call style; a persistent pool is later work.
        ComputeResult Decode(const unsigned char* jpegData, size_t jpegSize,
                              unsigned char** outRgb, int* outWidth, int* outHeight);

        void FreeDecoded(unsigned char* rgb);

        // Encodes an interleaved RGB8 buffer to JPEG bytes (preview/share
        // export path, docs/ARCHITECTURE.md SS4.5 - not the archival
        // render output, which stays a lossless format). quality: 0-100,
        // matches nvjpegEncoderParamsSetQuality's own range.
        ComputeResult Encode(const unsigned char* rgb, int width, int height, int quality,
                              std::vector<unsigned char>& outJpegBytes);

        const char* GetLastError() const;

    private:
        struct Impl;
        Impl* m_impl;
    };
}}
