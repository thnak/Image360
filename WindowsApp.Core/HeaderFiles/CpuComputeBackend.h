#pragma once

#include "IComputeBackend.h"
#include "CpuSimdDetect.h"

namespace WindowsApp::Core
{
    // CPU (AVX-512/AVX2/scalar, chosen at runtime via CpuSimdDetect)
    // implementation of Compute::IComputeBackend - the fallback for
    // machines with no compatible GPU. Ties together FastFeatureDetector/
    // BriefDescriptorExtractor/FeatureMatcher/GainColorOps and the tiered
    // MedianStack/WarpPerspective/BayerDemosaic kernels.
    class CpuComputeBackend : public Compute::IComputeBackend
    {
    public:
        CpuComputeBackend();
        ~CpuComputeBackend() override;

        Compute::ComputeResult Initialize() override;
        void Shutdown() override;
        bool IsInitialized() const override;
        Compute::GpuInfo GetGpuInfo() const override;
        const char* GetLastError() const override;

        Compute::ComputeResult WarpPerspective(
            const unsigned short* srcData, int srcW, int srcH,
            unsigned short* dstData, int dstW, int dstH,
            const float* homography, int offsetX, int offsetY) override;

        Compute::ComputeResult MedianStack(
            const unsigned short** inputs, int numInputs,
            unsigned short* output, int width, int height,
            float sigmaThreshold = 2.0f) override;

        Compute::ComputeResult ApplyGain(
            unsigned short* data, int numPixels, float gain) override;

        Compute::ComputeResult DemosaicBayer(
            const unsigned short* cfaData, int width, int height,
            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
            uint32_t filters, unsigned short* rgbOut) override;

        Compute::ComputeResult DetectAndDescribeFeatures(
            const unsigned char* rgbImage, int width, int height,
            Compute::FeaturePoint* outPoints, Compute::BriefDescriptor* outDescriptors, int* outCount, int maxPoints) override;

        Compute::ComputeResult MatchFeatures(
            const Compute::BriefDescriptor* descA, int countA,
            const Compute::BriefDescriptor* descB, int countB,
            Compute::MatchResult* outMatches, int* outMatchCount, int maxMatches,
            float ratioThreshold = 0.75f) override;

        Compute::ComputeResult ComputeLabStats(
            const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3]) override;

        Compute::ComputeResult ApplyReinhardColorTransfer(
            unsigned short* rgbInOut, int width, int height,
            const double srcMean[3], const double srcStd[3],
            const double refMean[3], const double refStd[3]) override;

        Compute::ComputeResult BlockMatchAlign(
            const unsigned short* refData, const unsigned short* srcData,
            int width, int height, int tileSize, int searchRadius,
            Compute::TileOffset* outOffsets, int tilesX, int tilesY) override;

        Compute::ComputeResult RobustMergeAccumulate(
            const unsigned short* const* frames, int numFrames,
            const Compute::TileOffset* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            float sigma, unsigned short* output) override;

        Compute::ComputeResult TileFftMerge(
            const unsigned short* const* frames, int numFrames,
            const Compute::TileOffset* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            float noiseVariance, unsigned short* output) override;

        Compute::ComputeResult StructureTensorKernelRegression(
            const unsigned short* const* frames, int numFrames,
            const Compute::TileOffsetF* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            int scaleFactor, float noiseVariance, unsigned short* output) override;

    private:
        bool m_initialized = false;
        SimdTier m_tier = SimdTier::Scalar;
        Compute::GpuInfo m_info;
        char m_lastError[512] = {};

        void SetError(const char* msg);
    };
}
