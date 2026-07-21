#pragma once

#include "ComputeTypes.h"
#include "IComputeBackend.h"

namespace WindowsApp { namespace Compute
{
    // Forward declaration for internal Vulkan state (instance/device/
    // pipelines/buffers) - kept out of this header so no Vulkan headers
    // leak into consumers, same pimpl approach as CudaPipeline/CudaContext.
    struct VulkanContext;

    // Vulkan compute implementation of IComputeBackend - the GPU path for
    // machines without a CUDA-capable device (e.g. AMD/Intel GPUs, the
    // win-amd CI runner). Real GLSL/SPIR-V compute shaders back
    // WarpPerspective/MedianStack/ApplyGain/DemosaicBayer (Stage 1
    // RawIngest / Stage 3 Render - the two heaviest, most embarrassingly
    // parallel stages). The remaining methods (feature detect/describe/
    // match, Lab stats, Reinhard color transfer) delegate to
    // WindowsApp::Core's portable CPU kernels - mirrors CudaPipeline
    // routing HomographyMath/LinearSolve through portable CPU code rather
    // than a dedicated GPU path (see IComputeBackend.h's own note on this
    // tradeoff).
    class VulkanPipeline : public IComputeBackend
    {
    public:
        VulkanPipeline();
        ~VulkanPipeline() override;

        VulkanPipeline(const VulkanPipeline&) = delete;
        VulkanPipeline& operator=(const VulkanPipeline&) = delete;

        ComputeResult Initialize() override;
        void Shutdown() override;
        bool IsInitialized() const override;
        GpuInfo GetGpuInfo() const override;
        const char* GetLastError() const override;

        ComputeResult WarpPerspective(
            const unsigned short* srcData, int srcW, int srcH,
            unsigned short* dstData, int dstW, int dstH,
            const float* homography, int offsetX, int offsetY) override;

        ComputeResult MedianStack(
            const unsigned short** inputs, int numInputs,
            unsigned short* output, int width, int height,
            float sigmaThreshold = 2.0f) override;

        ComputeResult ApplyGain(
            unsigned short* data, int numPixels, float gain) override;

        ComputeResult DemosaicBayer(
            const unsigned short* cfaData, int width, int height,
            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
            uint32_t filters, unsigned short* rgbOut) override;

        ComputeResult DetectAndDescribeFeatures(
            const unsigned char* rgbImage, int width, int height,
            FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints) override;

        ComputeResult MatchFeatures(
            const BriefDescriptor* descA, int countA,
            const BriefDescriptor* descB, int countB,
            MatchResult* outMatches, int* outMatchCount, int maxMatches,
            float ratioThreshold = 0.75f) override;

        ComputeResult ComputeLabStats(
            const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3]) override;

        ComputeResult ApplyReinhardColorTransfer(
            unsigned short* rgbInOut, int width, int height,
            const double srcMean[3], const double srcStd[3],
            const double refMean[3], const double refStd[3]) override;

        // Burst mode (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS3) -
        // NOT_SUPPORTED on this backend as of
        // docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md (CPU
        // only so far) - a real, tracked gap, not a silently-missing
        // feature. See CpuComputeBackend for the working implementation.
        ComputeResult BlockMatchAlign(
            const unsigned short* refData, const unsigned short* srcData,
            int width, int height, int tileSize, int searchRadius,
            TileOffset* outOffsets, int tilesX, int tilesY) override;

        ComputeResult RobustMergeAccumulate(
            const unsigned short* const* frames, int numFrames,
            const TileOffset* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            float sigma, unsigned short* output) override;

    private:
        VulkanContext* m_ctx;
        char m_lastError[512];

        void SetError(const char* msg);
    };
}}
