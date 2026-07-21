#include "pch.h"
#include "HeaderFiles/CpuComputeBackend.h"

#include "HeaderFiles/RgbToGray.h"
#include "HeaderFiles/FastFeatureDetector.h"
#include "HeaderFiles/BriefDescriptorExtractor.h"
#include "HeaderFiles/FeatureMatcher.h"
#include "HeaderFiles/GainColorOps.h"
#include "HeaderFiles/MedianStackKernels.h"
#include "HeaderFiles/WarpPerspectiveKernels.h"
#include "HeaderFiles/BayerDemosaicKernels.h"
#include "HeaderFiles/BlockMatchAlignKernel.h"
#include "HeaderFiles/RobustMergeKernel.h"
#include "HeaderFiles/TileFftMergeKernel.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace WindowsApp::Compute;

namespace WindowsApp::Core
{
    namespace
    {
        void QuerySystemMemory(size_t& totalBytes, size_t& freeBytes)
        {
#ifdef _WIN32
            MEMORYSTATUSEX status{};
            status.dwLength = sizeof(status);
            if (GlobalMemoryStatusEx(&status))
            {
                totalBytes = static_cast<size_t>(status.ullTotalPhys);
                freeBytes = static_cast<size_t>(status.ullAvailPhys);
                return;
            }
#else
            long pages = sysconf(_SC_PHYS_PAGES);
            long availPages = sysconf(_SC_AVPHYS_PAGES);
            long pageSize = sysconf(_SC_PAGE_SIZE);
            if (pages > 0 && pageSize > 0)
            {
                totalBytes = static_cast<size_t>(pages) * static_cast<size_t>(pageSize);
                freeBytes = static_cast<size_t>((availPages > 0) ? availPages : 0) * static_cast<size_t>(pageSize);
                return;
            }
#endif
            totalBytes = 0;
            freeBytes = 0;
        }
    }

    CpuComputeBackend::CpuComputeBackend() = default;
    CpuComputeBackend::~CpuComputeBackend() = default;

    void CpuComputeBackend::SetError(const char* msg)
    {
        std::snprintf(m_lastError, sizeof(m_lastError), "%s", msg);
    }

    ComputeResult CpuComputeBackend::Initialize()
    {
        m_tier = DetectCpuSimdTier();

        m_info = GpuInfo{};
        m_info.deviceId = -1;
        std::snprintf(m_info.name, sizeof(m_info.name), "%s", SimdTierName(m_tier));

        size_t totalBytes = 0, freeBytes = 0;
        QuerySystemMemory(totalBytes, freeBytes);
        m_info.totalMemory = totalBytes;
        m_info.freeMemory = freeBytes;
        m_info.computeMajor = 0;
        m_info.computeMinor = 0;
        m_info.maxThreadsPerBlock = 0;
        m_info.multiProcessorCount = static_cast<int>(std::thread::hardware_concurrency());
        m_info.hasTensorCores = false;

        m_initialized = true;
        return ComputeResult::SUCCESS;
    }

    void CpuComputeBackend::Shutdown()
    {
        m_initialized = false;
    }

    bool CpuComputeBackend::IsInitialized() const
    {
        return m_initialized;
    }

    GpuInfo CpuComputeBackend::GetGpuInfo() const
    {
        return m_info;
    }

    const char* CpuComputeBackend::GetLastError() const
    {
        return m_lastError;
    }

    ComputeResult CpuComputeBackend::WarpPerspective(
        const unsigned short* srcData, int srcW, int srcH,
        unsigned short* dstData, int dstW, int dstH,
        const float* homography, int offsetX, int offsetY)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!srcData || !dstData || !homography) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        switch (m_tier)
        {
        case SimdTier::Avx512:
            Kernels::Avx512::WarpPerspective(srcData, srcW, srcH, dstData, dstW, dstH, homography, offsetX, offsetY);
            break;
        case SimdTier::Avx2:
            Kernels::Avx2::WarpPerspective(srcData, srcW, srcH, dstData, dstW, dstH, homography, offsetX, offsetY);
            break;
        default:
            Kernels::Scalar::WarpPerspective(srcData, srcW, srcH, dstData, dstW, dstH, homography, offsetX, offsetY);
            break;
        }
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::MedianStack(
        const unsigned short** inputs, int numInputs,
        unsigned short* output, int width, int height,
        float sigmaThreshold)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!inputs || !output) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numInputs <= 0 || numInputs > 32) { SetError("numInputs must be 1-32."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        switch (m_tier)
        {
        case SimdTier::Avx512:
            Kernels::Avx512::MedianStack(inputs, numInputs, output, width, height, sigmaThreshold);
            break;
        case SimdTier::Avx2:
            Kernels::Avx2::MedianStack(inputs, numInputs, output, width, height, sigmaThreshold);
            break;
        default:
            Kernels::Scalar::MedianStack(inputs, numInputs, output, width, height, sigmaThreshold);
            break;
        }
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::ApplyGain(unsigned short* data, int numPixels, float gain)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!data) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numPixels <= 0) { SetError("Invalid pixel count."); return ComputeResult::INVALID_PARAM; }

        ApplyGainCpu(data, numPixels, gain);
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::DemosaicBayer(
        const unsigned short* cfaData, int width, int height,
        unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
        uint32_t filters, unsigned short* rgbOut)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!cfaData || !camMul || !rgbCam || !rgbOut) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        switch (m_tier)
        {
        case SimdTier::Avx512:
            Kernels::Avx512::DemosaicBayer(cfaData, width, height, blackLevel, camMul, rgbCam, filters, rgbOut);
            break;
        case SimdTier::Avx2:
            Kernels::Avx2::DemosaicBayer(cfaData, width, height, blackLevel, camMul, rgbCam, filters, rgbOut);
            break;
        default:
            Kernels::Scalar::DemosaicBayer(cfaData, width, height, blackLevel, camMul, rgbCam, filters, rgbOut);
            break;
        }
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::DetectAndDescribeFeatures(
        const unsigned char* rgbImage, int width, int height,
        FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgbImage || !outPoints || !outDescriptors || !outCount) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0 || maxPoints <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        std::vector<unsigned char> gray(static_cast<size_t>(width) * height);
        ConvertRgbToGray(rgbImage, width, height, gray.data());

        int detectedCount = 0;
        DetectFastCorners(gray.data(), width, height, outPoints, &detectedCount, maxPoints);
        int clampedCount = (std::min)(detectedCount, maxPoints);

        if (clampedCount > 0)
        {
            ExtractBriefDescriptors(gray.data(), width, height, outPoints, clampedCount, outDescriptors);
        }

        *outCount = clampedCount;
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::MatchFeatures(
        const BriefDescriptor* descA, int countA,
        const BriefDescriptor* descB, int countB,
        MatchResult* outMatches, int* outMatchCount, int maxMatches,
        float ratioThreshold)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!descA || !descB || !outMatches || !outMatchCount) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }

        int rawCount = 0;
        MatchFeaturesBruteForce(descA, countA, descB, countB, outMatches, &rawCount, maxMatches, ratioThreshold);
        *outMatchCount = (std::min)(rawCount, maxMatches);
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::ComputeLabStats(
        const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3])
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgb || !outMean || !outStd) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        ComputeLabStatsCpu(rgb, width, height, outMean, outStd);
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::ApplyReinhardColorTransfer(
        unsigned short* rgbInOut, int width, int height,
        const double srcMean[3], const double srcStd[3],
        const double refMean[3], const double refStd[3])
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgbInOut || !srcMean || !srcStd || !refMean || !refStd) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        ApplyReinhardColorTransferCpu(rgbInOut, width, height, srcMean, srcStd, refMean, refStd);
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::BlockMatchAlign(
        const unsigned short* refData, const unsigned short* srcData,
        int width, int height, int tileSize, int searchRadius,
        TileOffset* outOffsets, int tilesX, int tilesY)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!refData || !srcData || !outOffsets) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0 || tileSize <= 0 || searchRadius < 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }
        if (tilesX <= 0 || tilesY <= 0) { SetError("Invalid tile grid."); return ComputeResult::INVALID_PARAM; }

        Kernels::BlockMatchAlign(refData, srcData, width, height, tileSize, searchRadius, outOffsets, tilesX, tilesY);
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::RobustMergeAccumulate(
        const unsigned short* const* frames, int numFrames,
        const TileOffset* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        float sigma, unsigned short* output)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!frames || !output) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numFrames <= 0) { SetError("numFrames must be >= 1."); return ComputeResult::INVALID_PARAM; }
        if (numFrames > 1 && !perFrameOffsets) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0 || tileSize <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }
        if (tilesX <= 0 || tilesY <= 0) { SetError("Invalid tile grid."); return ComputeResult::INVALID_PARAM; }
        if (sigma <= 0.0f) { SetError("sigma must be positive."); return ComputeResult::INVALID_PARAM; }

        Kernels::RobustMergeAccumulate(frames, numFrames, perFrameOffsets, width, height, tileSize, tilesX, tilesY, sigma, output);
        return ComputeResult::SUCCESS;
    }

    ComputeResult CpuComputeBackend::TileFftMerge(
        const unsigned short* const* frames, int numFrames,
        const TileOffset* const* perFrameOffsets,
        int width, int height, int tileSize, int tilesX, int tilesY,
        float noiseVariance, unsigned short* output)
    {
        if (!m_initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!frames || !output) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numFrames <= 0) { SetError("numFrames must be >= 1."); return ComputeResult::INVALID_PARAM; }
        if (numFrames > 1 && !perFrameOffsets) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0 || tileSize <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }
        if (tilesX <= 0 || tilesY <= 0) { SetError("Invalid tile grid."); return ComputeResult::INVALID_PARAM; }
        if (noiseVariance < 0.0f) { SetError("noiseVariance must be non-negative."); return ComputeResult::INVALID_PARAM; }

        if (!Kernels::TileFftMerge(frames, numFrames, perFrameOffsets, width, height, tileSize, tilesX, tilesY, noiseVariance, output))
        {
            SetError("tileSize must be a power of two (TileFftMerge is a radix-2 FFT).");
            return ComputeResult::INVALID_PARAM;
        }
        return ComputeResult::SUCCESS;
    }
}
