#pragma once

#if defined(_MSC_VER)
#ifdef WINDOWSAPPCOMPUTE_EXPORTS
#define COMPUTE_API __declspec(dllexport)
#else
#define COMPUTE_API __declspec(dllimport)
#endif
#elif defined(__GNUC__)
#define COMPUTE_API __attribute__((visibility("default")))
#else
#define COMPUTE_API
#endif

#include <cstdint>
#include <cstddef>

#include "ComputeTypes.h"
#include "IComputeBackend.h"

namespace WindowsApp { namespace Compute
{
    // Forward declaration for internal CUDA state
    struct CudaContext;

    class COMPUTE_API CudaPipeline : public IComputeBackend
    {
    public:
        CudaPipeline();
        ~CudaPipeline();

        // Disable copy
        CudaPipeline(const CudaPipeline&) = delete;
        CudaPipeline& operator=(const CudaPipeline&) = delete;

        // Initialize CUDA context, query GPU info
        ComputeResult Initialize() override;
        void Shutdown() override;
        bool IsInitialized() const override;

        // Get GPU info after initialization
        GpuInfo GetGpuInfo() const override;

        // =====================================================================
        // Kernel 1: Perspective Warp (backward mapping with inverse homography)
        // =====================================================================
        // srcData: source image pixels (RGB48, unsigned short per channel)
        // srcW, srcH: source dimensions
        // dstData: output buffer (pre-allocated)
        // dstW, dstH: output dimensions
        // homography: 3x3 matrix (row-major, 9 floats)
        // offsetX, offsetY: translation offset in source space
        ComputeResult WarpPerspective(
            const unsigned short* srcData, int srcW, int srcH,
            unsigned short* dstData, int dstW, int dstH,
            const float* homography, int offsetX, int offsetY) override;

        // =====================================================================
        // Kernel 2: Median Stack with Sigma-Clipping
        // =====================================================================
        // inputs: array of pointers to input images (all same size)
        // numInputs: number of input images
        // output: pre-allocated output buffer
        // width, height: image dimensions
        // sigmaThreshold: sigma multiplier for outlier rejection (default 2.0)
        ComputeResult MedianStack(
            const unsigned short** inputs, int numInputs,
            unsigned short* output, int width, int height,
            float sigmaThreshold = 2.0f) override;

        // =====================================================================
        // Kernel 3: Multi-Band Blending (Laplacian Pyramid)
        // =====================================================================
        // imgA, imgB: two images to blend (same size)
        // output: pre-allocated output buffer
        // width, height: image dimensions
        // numBands: number of pyramid levels (default 5)
        ComputeResult MultiBandBlend(
            const unsigned short* imgA, const unsigned short* imgB,
            unsigned short* output, int width, int height,
            int numBands = 5);

        // =====================================================================
        // Kernel 4: Gain Compensation (apply per-pixel gain)
        // =====================================================================
        // data: image data (in-place modification)
        // numPixels: total pixel count (width * height * channels)
        // gain: multiplicative gain factor
        ComputeResult ApplyGain(
            unsigned short* data, int numPixels, float gain) override;

        // =====================================================================
        // Kernel 5: GPU Demosaic (RawIngest, docs/ARCHITECTURE.md SS4.1)
        // =====================================================================
        // cfaData: raw sensor plane, one sample per pixel (width*height)
        // camMul: per-channel white balance multipliers (4 slots, LibRaw
        //   convention - index 3 is the second green channel)
        // rgbCam: camera RGB -> sRGB matrix, row-major 3x4
        // filters: LibRaw's CFA pattern encoding
        // rgbOut: pre-allocated, width*height*3 (RGB48)
        ComputeResult DemosaicBayer(
            const unsigned short* cfaData, int width, int height,
            unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
            uint32_t filters, unsigned short* rgbOut) override;

        // =====================================================================
        // Align: FAST detect + BRIEF describe (docs/ARCHITECTURE.md SS4.2)
        // =====================================================================
        // rgbImage: interleaved RGB8, width*height*3 (e.g. from
        //   NvJpegCodec::Decode). outPoints/outDescriptors: caller-allocated,
        //   maxPoints capacity. outCount: actual detections (<= maxPoints).
        ComputeResult DetectAndDescribeFeatures(
            const unsigned char* rgbImage, int width, int height,
            FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints) override;

        // =====================================================================
        // Align: Brute-force descriptor matching (docs/ARCHITECTURE.md SS4.2)
        // =====================================================================
        // One thread per descriptor in A finds its best/second-best match
        // in B (Hamming distance); accepts via Lowe's ratio test. Returns
        // at most maxMatches results in outMatches.
        ComputeResult MatchFeatures(
            const BriefDescriptor* descA, int countA,
            const BriefDescriptor* descB, int countB,
            MatchResult* outMatches, int* outMatchCount, int maxMatches,
            float ratioThreshold = 0.75f) override;

        // =====================================================================
        // Optimize: Reinhard color transfer (docs/ARCHITECTURE.md SS4.3)
        // =====================================================================
        // outMean/outStd: 3 doubles (L, a, b channels).
        ComputeResult ComputeLabStats(
            const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3]) override;

        // rgbInOut: width*height*3, modified in place.
        ComputeResult ApplyReinhardColorTransfer(
            unsigned short* rgbInOut, int width, int height,
            const double srcMean[3], const double srcStd[3],
            const double refMean[3], const double refStd[3]) override;

        // =====================================================================
        // Burst mode (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS3) - NOT_SUPPORTED
        // on this backend as of
        // docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md (CPU
        // only so far) - a real, tracked gap, not a silently-missing
        // feature. See CpuComputeBackend for the working implementation.
        // =====================================================================
        ComputeResult BlockMatchAlign(
            const unsigned short* refData, const unsigned short* srcData,
            int width, int height, int tileSize, int searchRadius,
            TileOffset* outOffsets, int tilesX, int tilesY) override;

        ComputeResult RobustMergeAccumulate(
            const unsigned short* const* frames, int numFrames,
            const TileOffset* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            float sigma, unsigned short* output) override;

        // NOT_SUPPORTED on this backend as of docs/superpowers/plans/
        // 2026-07-21-hdrplus-tile-fft-merge.md (CPU only so far).
        ComputeResult TileFftMerge(
            const unsigned short* const* frames, int numFrames,
            const TileOffset* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            float noiseVariance, unsigned short* output) override;

        // NOT_SUPPORTED on this backend as of docs/superpowers/plans/
        // 2026-07-21-superres-structure-tensor-merge.md (CPU only so far).
        ComputeResult StructureTensorKernelRegression(
            const unsigned short* const* frames, int numFrames,
            const TileOffsetF* const* perFrameOffsets,
            int width, int height, int tileSize, int tilesX, int tilesY,
            int scaleFactor, float noiseVariance, unsigned short* output) override;

        // =====================================================================
        // Tensor Core Operations (requires SM 7.0+)
        // =====================================================================

        // Batch matrix multiply using tensor cores (FP16 compute, FP32 accumulate)
        // A: batchA_M x batchA_K per batch, B: batchA_K x batchB_N per batch
        // C: batchA_M x batchB_N per batch (output)
        // All matrices in row-major layout, one after another for each batch.
        ComputeResult TensorBatchMatMul(
            const float* A, const float* B, float* C,
            int batchA_M, int batchA_K, int batchB_N,
            int batchSize);

        // Estimate homography from point correspondences using tensor cores
        // pointPairs: [x0,y0,x0',y0', x1,y1,x1',y1', ...] (float, 4 per pair)
        // homography: output 3x3 matrix (float, row-major, 9 elements)
        // numPairs: number of correspondences (minimum 4)
        ComputeResult TensorEstimateHomography(
            const float* pointPairs, float* homography, int numPairs);

        // Solve normal equations JtJ * delta = -Jtr using tensor cores
        // For Levenberg-Marquardt bundle adjustment step.
        // J: Jacobian (numResiduals x numParams), row-major float
        // r: residual vector (numResiduals)
        // delta: output parameter update (numParams)
        // lambda: LM damping factor
        ComputeResult TensorSolveNormalEquations(
            const float* J, const float* r, float* delta,
            int numResiduals, int numParams, float lambda);

        // Get last error string
        const char* GetLastError() const override;

    private:
        CudaContext* m_ctx;
        char m_lastError[512];

        void SetError(const char* msg);
    };
    }
}
