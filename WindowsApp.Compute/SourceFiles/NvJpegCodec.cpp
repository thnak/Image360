#include "HeaderFiles/NvJpegCodec.h"
#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <cstring>

namespace WindowsApp { namespace Compute
{
    struct NvJpegCodec::Impl
    {
        nvjpegHandle_t handle = nullptr;
        nvjpegJpegState_t state = nullptr;
        bool initialized = false;
        char lastError[512] = {};

        void SetError(const char* msg)
        {
            strncpy_s(lastError, sizeof(lastError), msg, _TRUNCATE);
        }
    };

    NvJpegCodec::NvJpegCodec()
        : m_impl(new Impl())
    {
    }

    NvJpegCodec::~NvJpegCodec()
    {
        Shutdown();
        delete m_impl;
    }

    ComputeResult NvJpegCodec::Initialize()
    {
        if (nvjpegCreateSimple(&m_impl->handle) != NVJPEG_STATUS_SUCCESS)
        {
            m_impl->SetError("nvjpegCreateSimple failed.");
            return ComputeResult::CUDA_ERROR;
        }

        if (nvjpegJpegStateCreate(m_impl->handle, &m_impl->state) != NVJPEG_STATUS_SUCCESS)
        {
            m_impl->SetError("nvjpegJpegStateCreate failed.");
            nvjpegDestroy(m_impl->handle);
            m_impl->handle = nullptr;
            return ComputeResult::CUDA_ERROR;
        }

        m_impl->initialized = true;
        return ComputeResult::SUCCESS;
    }

    void NvJpegCodec::Shutdown()
    {
        if (m_impl->initialized)
        {
            if (m_impl->state) nvjpegJpegStateDestroy(m_impl->state);
            if (m_impl->handle) nvjpegDestroy(m_impl->handle);
            m_impl->state = nullptr;
            m_impl->handle = nullptr;
            m_impl->initialized = false;
        }
    }

    ComputeResult NvJpegCodec::Decode(const unsigned char* jpegData, size_t jpegSize,
                                       unsigned char** outRgb, int* outWidth, int* outHeight)
    {
        if (!m_impl->initialized)
        {
            m_impl->SetError("Not initialized.");
            return ComputeResult::CUDA_ERROR;
        }
        if (!jpegData || !outRgb || !outWidth || !outHeight)
        {
            m_impl->SetError("Null pointer argument.");
            return ComputeResult::INVALID_PARAM;
        }

        int nComponents = 0;
        nvjpegChromaSubsampling_t subsampling{};
        int widths[NVJPEG_MAX_COMPONENT] = {};
        int heights[NVJPEG_MAX_COMPONENT] = {};

        if (nvjpegGetImageInfo(
                m_impl->handle, jpegData, jpegSize,
                &nComponents, &subsampling, widths, heights) != NVJPEG_STATUS_SUCCESS)
        {
            m_impl->SetError("nvjpegGetImageInfo failed.");
            return ComputeResult::CUDA_ERROR;
        }

        int width = widths[0];
        int height = heights[0];
        size_t pitch = static_cast<size_t>(width) * 3;
        size_t bufferSize = pitch * static_cast<size_t>(height);

        nvjpegImage_t destination = {};
        if (cudaMalloc(reinterpret_cast<void**>(&destination.channel[0]), bufferSize) != cudaSuccess)
        {
            m_impl->SetError("cudaMalloc failed for decoded RGB buffer.");
            return ComputeResult::OUT_OF_MEMORY;
        }
        destination.pitch[0] = static_cast<unsigned int>(pitch);

        nvjpegStatus_t decodeStatus = nvjpegDecode(
            m_impl->handle, m_impl->state, jpegData, jpegSize,
            NVJPEG_OUTPUT_RGBI, &destination, nullptr);
        if (decodeStatus != NVJPEG_STATUS_SUCCESS)
        {
            cudaFree(destination.channel[0]);
            m_impl->SetError("nvjpegDecode failed.");
            return ComputeResult::CUDA_ERROR;
        }

        cudaDeviceSynchronize();

        unsigned char* hostRgb = new unsigned char[bufferSize];
        cudaError_t copyErr = cudaMemcpy(hostRgb, destination.channel[0], bufferSize, cudaMemcpyDeviceToHost);
        cudaFree(destination.channel[0]);

        if (copyErr != cudaSuccess)
        {
            delete[] hostRgb;
            m_impl->SetError("cudaMemcpy failed for decoded RGB buffer.");
            return ComputeResult::CUDA_ERROR;
        }

        *outRgb = hostRgb;
        *outWidth = width;
        *outHeight = height;
        return ComputeResult::SUCCESS;
    }

    void NvJpegCodec::FreeDecoded(unsigned char* rgb)
    {
        delete[] rgb;
    }

    ComputeResult NvJpegCodec::Encode(const unsigned char* rgb, int width, int height, int quality,
                                       std::vector<unsigned char>& outJpegBytes)
    {
        if (!m_impl->initialized)
        {
            m_impl->SetError("Not initialized.");
            return ComputeResult::CUDA_ERROR;
        }
        if (!rgb || width <= 0 || height <= 0)
        {
            m_impl->SetError("Invalid argument.");
            return ComputeResult::INVALID_PARAM;
        }

        nvjpegEncoderParams_t params = nullptr;
        if (nvjpegEncoderParamsCreate(m_impl->handle, &params, nullptr) != NVJPEG_STATUS_SUCCESS)
        {
            m_impl->SetError("nvjpegEncoderParamsCreate failed.");
            return ComputeResult::CUDA_ERROR;
        }
        nvjpegEncoderParamsSetQuality(params, quality, nullptr);
        nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_420, nullptr);

        nvjpegEncoderState_t state = nullptr;
        if (nvjpegEncoderStateCreate(m_impl->handle, &state, nullptr) != NVJPEG_STATUS_SUCCESS)
        {
            m_impl->SetError("nvjpegEncoderStateCreate failed.");
            nvjpegEncoderParamsDestroy(params);
            return ComputeResult::CUDA_ERROR;
        }

        size_t pitch = static_cast<size_t>(width) * 3;
        size_t bufferSize = pitch * static_cast<size_t>(height);

        nvjpegImage_t source = {};
        if (cudaMalloc(reinterpret_cast<void**>(&source.channel[0]), bufferSize) != cudaSuccess)
        {
            m_impl->SetError("cudaMalloc failed for encode source buffer.");
            nvjpegEncoderStateDestroy(state);
            nvjpegEncoderParamsDestroy(params);
            return ComputeResult::OUT_OF_MEMORY;
        }
        source.pitch[0] = static_cast<unsigned int>(pitch);

        if (cudaMemcpy(source.channel[0], rgb, bufferSize, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            cudaFree(source.channel[0]);
            nvjpegEncoderStateDestroy(state);
            nvjpegEncoderParamsDestroy(params);
            m_impl->SetError("cudaMemcpy failed for encode source buffer.");
            return ComputeResult::CUDA_ERROR;
        }

        nvjpegStatus_t encodeStatus = nvjpegEncodeImage(
            m_impl->handle, state, params, &source, NVJPEG_INPUT_RGBI, width, height, nullptr);

        cudaFree(source.channel[0]);

        if (encodeStatus != NVJPEG_STATUS_SUCCESS)
        {
            nvjpegEncoderStateDestroy(state);
            nvjpegEncoderParamsDestroy(params);
            m_impl->SetError("nvjpegEncodeImage failed.");
            return ComputeResult::CUDA_ERROR;
        }

        size_t bitstreamSize = 0;
        nvjpegEncodeRetrieveBitstream(m_impl->handle, state, nullptr, &bitstreamSize, nullptr);

        outJpegBytes.resize(bitstreamSize);
        nvjpegStatus_t retrieveStatus = nvjpegEncodeRetrieveBitstream(
            m_impl->handle, state, outJpegBytes.data(), &bitstreamSize, nullptr);

        nvjpegEncoderStateDestroy(state);
        nvjpegEncoderParamsDestroy(params);

        if (retrieveStatus != NVJPEG_STATUS_SUCCESS)
        {
            m_impl->SetError("nvjpegEncodeRetrieveBitstream failed.");
            return ComputeResult::CUDA_ERROR;
        }

        return ComputeResult::SUCCESS;
    }

    const char* NvJpegCodec::GetLastError() const
    {
        return m_impl->lastError;
    }
}}
