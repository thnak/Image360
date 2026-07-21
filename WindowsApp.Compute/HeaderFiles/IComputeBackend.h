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

        // Burst-mode alignment (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.1,
        // SS3) - shared by all burst modes (MFNR/HDR+/Night Sight/Super
        // Res Zoom). Non-overlapping tileSize x tileSize grid over
        // refData/srcData (edge tiles clipped to width/height). For each
        // tile, brute-force integer-pixel SAD search in srcData within
        // [-searchRadius, +searchRadius] of the tile's position in
        // refData, picking the minimum-SAD offset (ties broken toward the
        // smallest |dx|+|dy|). outOffsets: caller-allocated,
        // tilesX*tilesY entries, row-major (tilesX = ceil(width/tileSize),
        // same for Y). refData/srcData: RGB48, same width/height.
        // CPU-only as of the plan above - CUDA/Vulkan return
        // NOT_SUPPORTED (a tracked gap, not silently missing).
        virtual ComputeResult BlockMatchAlign(
            const unsigned short* refData, const unsigned short* srcData,
            int width, int height, int tileSize, int searchRadius,
            TileOffset* outOffsets, int tilesX, int tilesY) = 0;

        // MFNR's merge (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.3) -
        // reference-frame-relative Gaussian-weighted merge, NOT HDR+'s
        // FFT/Wiener-shrinkage merge or Super-Res-Zoom's kernel-regression
        // merge (those are separate, later ops per SS2.3's correction).
        // frames[0] is the reference (implicit zero offset, weight 1.0 -
        // must NOT have a corresponding perFrameOffsets entry);
        // frames[1..numFrames) are aligned via perFrameOffsets[k-1]
        // (BlockMatchAlign's output against frames[0], tilesX*tilesY
        // entries each) - integer-pixel sampled (TileOffset has no
        // fractional part in this phase; a future sub-pixel-refined
        // BlockMatchAlign could add bilinear sampling here without an
        // interface change), weighted by
        // exp(-(sample-reference)^2 / (2*sigma^2)). A sample landing
        // outside [0,width)x[0,height) after applying its tile's offset is
        // excluded from that pixel's average entirely (never contributes
        // a spurious zero - see RenderExecutor's CombineIgnoringGaps for
        // the same "gap != black" principle). output: caller-allocated
        // width*height*3 (the reference's own dimensions). CPU-only as of
        // the plan above - CUDA/Vulkan return NOT_SUPPORTED.
        virtual ComputeResult RobustMergeAccumulate(
            const unsigned short* const* frames, int numFrames,
            const TileOffset* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            float sigma, unsigned short* output) = 0;
    };
}}
