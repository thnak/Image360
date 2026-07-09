#include "pch.h"
#include "HeaderFiles/JpegCodec.h"

#include <turbojpeg.h>
#include <cstdio>
#include <cstring>

using namespace WindowsApp::Compute;

namespace WindowsApp::Core
{
    namespace
    {
        // turbojpeg handles are only thread-safe when each thread uses its
        // own handle - AlignExecutor/OptimizeExecutor's feature-extraction
        // and gain tasks run concurrently (TaskScheduler dispatches
        // multiple "image" tasks per stage via std::async), all sharing
        // one JpegCodec instance, so Decode()/Encode() must not reuse a
        // single member handle across calls. tjInitDecompress/
        // tjInitCompress are cheap, so a fresh handle per call (freed via
        // this RAII wrapper on every return path) is simpler and safer
        // than serializing all JPEG work behind a mutex.
        struct ScopedTjHandle
        {
            tjhandle handle;
            explicit ScopedTjHandle(tjhandle h) : handle(h) {}
            ~ScopedTjHandle() { if (handle) tjDestroy(handle); }
            ScopedTjHandle(const ScopedTjHandle&) = delete;
            ScopedTjHandle& operator=(const ScopedTjHandle&) = delete;
        };
    }

    struct JpegCodec::Impl
    {
        bool initialized = false;
        char lastError[512] = {};
    };

    JpegCodec::JpegCodec() : m_impl(new Impl()) {}

    JpegCodec::~JpegCodec()
    {
        Shutdown();
        delete m_impl;
    }

    void JpegCodec::SetError(const char* msg)
    {
        std::snprintf(m_impl->lastError, sizeof(m_impl->lastError), "%s", msg);
    }

    ComputeResult JpegCodec::Initialize()
    {
        // Fail-fast liveness probe only - Decode/Encode each create their
        // own short-lived handle (see ScopedTjHandle above).
        ScopedTjHandle compressor(tjInitCompress());
        ScopedTjHandle decompressor(tjInitDecompress());
        if (!compressor.handle || !decompressor.handle)
        {
            SetError("Failed to initialize libjpeg-turbo compressor/decompressor.");
            return ComputeResult::CUDA_ERROR;
        }
        m_impl->initialized = true;
        return ComputeResult::SUCCESS;
    }

    void JpegCodec::Shutdown()
    {
        m_impl->initialized = false;
    }

    ComputeResult JpegCodec::Decode(const unsigned char* jpegData, size_t jpegSize,
                                     unsigned char** outRgb, int* outWidth, int* outHeight)
    {
        if (!m_impl->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!jpegData || !outRgb || !outWidth || !outHeight) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }

        ScopedTjHandle decompressor(tjInitDecompress());
        if (!decompressor.handle) { SetError("Out of memory."); return ComputeResult::OUT_OF_MEMORY; }

        int width = 0, height = 0, subsamp = 0, colorspace = 0;
        if (tjDecompressHeader3(decompressor.handle, jpegData, static_cast<unsigned long>(jpegSize),
                                 &width, &height, &subsamp, &colorspace) != 0)
        {
            SetError(tjGetErrorStr2(decompressor.handle));
            return ComputeResult::INVALID_PARAM;
        }

        unsigned char* rgb = tjAlloc(width * height * 3);
        if (!rgb) { SetError("Out of memory."); return ComputeResult::OUT_OF_MEMORY; }

        if (tjDecompress2(decompressor.handle, jpegData, static_cast<unsigned long>(jpegSize),
                           rgb, width, 0, height, TJPF_RGB, 0) != 0)
        {
            SetError(tjGetErrorStr2(decompressor.handle));
            tjFree(rgb);
            return ComputeResult::CUDA_ERROR;
        }

        *outRgb = rgb;
        *outWidth = width;
        *outHeight = height;
        return ComputeResult::SUCCESS;
    }

    void JpegCodec::FreeDecoded(unsigned char* rgb)
    {
        if (rgb) tjFree(rgb);
    }

    ComputeResult JpegCodec::Encode(const unsigned char* rgb, int width, int height, int quality,
                                     std::vector<unsigned char>& outJpegBytes)
    {
        if (!m_impl->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgb) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        ScopedTjHandle compressor(tjInitCompress());
        if (!compressor.handle) { SetError("Out of memory."); return ComputeResult::OUT_OF_MEMORY; }

        unsigned char* jpegBuf = nullptr;
        unsigned long jpegSize = 0;

        int rc = tjCompress2(compressor.handle, rgb, width, 0, height, TJPF_RGB,
                              &jpegBuf, &jpegSize, TJSAMP_420, quality, 0);
        if (rc != 0)
        {
            SetError(tjGetErrorStr2(compressor.handle));
            if (jpegBuf) tjFree(jpegBuf);
            return ComputeResult::CUDA_ERROR;
        }

        outJpegBytes.assign(jpegBuf, jpegBuf + jpegSize);
        tjFree(jpegBuf);
        return ComputeResult::SUCCESS;
    }

    const char* JpegCodec::GetLastError() const
    {
        return m_impl->lastError;
    }
}
