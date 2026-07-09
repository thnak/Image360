#pragma once

#include "ComputeTypes.h"

namespace WindowsApp { namespace Compute
{
    // Backend-agnostic compute surface - exactly the operations Core's
    // executors call today. Implemented by CudaPipeline (GPU) and
    // CpuComputeBackend (AVX-512/AVX2/scalar, chosen at runtime). Excludes
    // CudaPipeline's TensorEstimateHomography/TensorSolveNormalEquations
    // (replaced by WindowsApp.Core's own portable HomographyMath/LinearSolve
    // - RANSAC/bundle-adjustment orchestration already does its expensive
    // work on the host and only offloaded a tiny solve step, so routing that
    // through a backend added latency without benefit) and
    // MultiBandBlend/TensorBatchMatMul (unused by any executor today).
    class IComputeBackend
    {
    public:
        virtual ~IComputeBackend() = default;

        virtual ComputeResult Initialize() = 0;
        virtual void Shutdown() = 0;
        virtual bool IsInitialized() const = 0;
        virtual GpuInfo GetGpuInfo() const = 0;
        virtual const char* GetLastError() const = 0;

        // Perspective warp (backward mapping, inverse homography).
        // srcData/dstData: RGB48 (unsigned short per channel).
        virtual ComputeResult WarpPerspective(
            const unsigned short* srcData, int srcW, int srcH,
            unsigned short* dstData, int dstW, int dstH,
            const float* homography, int offsetX, int offsetY) = 0;

        // Sigma-clipped median stack across up to kMaxContributors inputs.
        virtual ComputeResult MedianStack(
            const unsigned short** inputs, int numInputs,
            unsigned short* output, int width, int height,
            float sigmaThreshold = 2.0f) = 0;

        // In-place multiplicative gain.
        virtual ComputeResult ApplyGain(
            unsigned short* data, int numPixels, float gain) = 0;

        // Bayer demosaic (RawIngest). cfaData: one sample/pixel; rgbOut:
        // pre-allocated width*height*3 (RGB48).
        virtual ComputeResult DemosaicBayer(
            const unsigned short* cfaData, int width, int height,
            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
            uint32_t filters, unsigned short* rgbOut) = 0;

        // FAST detect + BRIEF describe. rgbImage: interleaved RGB8.
        // outPoints/outDescriptors: caller-allocated, maxPoints capacity.
        virtual ComputeResult DetectAndDescribeFeatures(
            const unsigned char* rgbImage, int width, int height,
            FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints) = 0;

        // Brute-force Hamming-distance descriptor matching with Lowe's
        // ratio test.
        virtual ComputeResult MatchFeatures(
            const BriefDescriptor* descA, int countA,
            const BriefDescriptor* descB, int countB,
            MatchResult* outMatches, int* outMatchCount, int maxMatches,
            float ratioThreshold = 0.75f) = 0;

        // Reinhard color transfer. outMean/outStd: 3 doubles (L, a, b).
        virtual ComputeResult ComputeLabStats(
            const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3]) = 0;

        // rgbInOut: width*height*3, modified in place.
        virtual ComputeResult ApplyReinhardColorTransfer(
            unsigned short* rgbInOut, int width, int height,
            const double srcMean[3], const double srcStd[3],
            const double refMean[3], const double refStd[3]) = 0;
    };
}}
