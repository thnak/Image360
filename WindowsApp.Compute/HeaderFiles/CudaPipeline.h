#pragma once

#ifdef WINDOWSAPPCOMPUTE_EXPORTS
#define COMPUTE_API __declspec(dllexport)
#else
#define COMPUTE_API __declspec(dllimport)
#endif

#include <cstdint>
#include <cstddef>

namespace WindowsApp { namespace Compute
{
    // GPU device information
    struct GpuInfo
    {
        int deviceId = -1;
        char name[256] = {};
        size_t totalMemory = 0;
        size_t freeMemory = 0;
        int computeMajor = 0;
        int computeMinor = 0;
        int maxThreadsPerBlock = 0;
        int multiProcessorCount = 0;
        bool hasTensorCores = false;  // Volta (SM 7.0+) or Ampere (SM 8.0+)
    };

    // Result codes
    enum class ComputeResult
    {
        SUCCESS = 0,
        NO_GPU = 1,
        INVALID_PARAM = 2,
        OUT_OF_MEMORY = 3,
        KERNEL_LAUNCH_FAILED = 4,
        CUDA_ERROR = 5
    };

    // Forward declaration for internal CUDA state
    struct CudaContext;

    class COMPUTE_API CudaPipeline
    {
    public:
        CudaPipeline();
        ~CudaPipeline();

        // Disable copy
        CudaPipeline(const CudaPipeline&) = delete;
        CudaPipeline& operator=(const CudaPipeline&) = delete;

        // Initialize CUDA context, query GPU info
        ComputeResult Initialize();
        void Shutdown();
        bool IsInitialized() const;

        // Get GPU info after initialization
        GpuInfo GetGpuInfo() const;

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
            const float* homography, int offsetX, int offsetY);

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
            float sigmaThreshold = 2.0f);

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
            unsigned short* data, int numPixels, float gain);

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
            uint32_t filters, unsigned short* rgbOut);

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
        const char* GetLastError() const;

    private:
        CudaContext* m_ctx;
        char m_lastError[512];

        void SetError(const char* msg);
    };
    }
}
