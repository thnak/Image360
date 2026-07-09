#pragma once

#include "IImageCodec.h"

namespace WindowsApp::Core
{
    // CPU JPEG codec via vendored libjpeg-turbo (turbojpeg API) - the
    // fallback for machines with no GPU/nvJPEG. Decode/Encode semantics
    // match NvJpegCodec exactly (interleaved RGB8 on the plain side).
    class JpegCodec : public Compute::IImageCodec
    {
    public:
        JpegCodec();
        ~JpegCodec() override;
        JpegCodec(const JpegCodec&) = delete;
        JpegCodec& operator=(const JpegCodec&) = delete;

        Compute::ComputeResult Initialize() override;
        void Shutdown() override;

        Compute::ComputeResult Decode(const unsigned char* jpegData, size_t jpegSize,
                                       unsigned char** outRgb, int* outWidth, int* outHeight) override;

        void FreeDecoded(unsigned char* rgb) override;

        Compute::ComputeResult Encode(const unsigned char* rgb, int width, int height, int quality,
                                       std::vector<unsigned char>& outJpegBytes) override;

        const char* GetLastError() const override;

    private:
        struct Impl;
        Impl* m_impl;

        void SetError(const char* msg);
    };
}
