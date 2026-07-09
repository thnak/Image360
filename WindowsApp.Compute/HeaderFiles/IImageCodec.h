#pragma once

#include "ComputeTypes.h"
#include <cstddef>
#include <vector>

namespace WindowsApp { namespace Compute
{
    // Backend-agnostic JPEG codec surface, mirroring NvJpegCodec's public
    // API 1:1. Implemented by NvJpegCodec (nvJPEG, GPU) and JpegCodec
    // (libjpeg-turbo, CPU). Every current call site already converts
    // PixelBuffer's 16-bit data to 8-bit before crossing this boundary and
    // back after, so no 8/16-bit conversion belongs inside the codec itself.
    class IImageCodec
    {
    public:
        virtual ~IImageCodec() = default;

        virtual ComputeResult Initialize() = 0;
        virtual void Shutdown() = 0;

        // Decodes a JPEG byte buffer to interleaved RGB8, allocating
        // outRgb (caller frees via FreeDecoded).
        virtual ComputeResult Decode(const unsigned char* jpegData, size_t jpegSize,
                                     unsigned char** outRgb, int* outWidth, int* outHeight) = 0;

        virtual void FreeDecoded(unsigned char* rgb) = 0;

        // Encodes an interleaved RGB8 buffer to JPEG bytes. quality: 0-100.
        virtual ComputeResult Encode(const unsigned char* rgb, int width, int height, int quality,
                                     std::vector<unsigned char>& outJpegBytes) = 0;

        virtual const char* GetLastError() const = 0;
    };
}}
