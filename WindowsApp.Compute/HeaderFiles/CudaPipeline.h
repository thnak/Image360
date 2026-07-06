#pragma once

#ifdef WINDOWSAPPCOMPUTE_EXPORTS
#define COMPUTE_API __declspec(dllexport)
#else
#define COMPUTE_API __declspec(dllimport)
#endif

#include <cstdint>
#include <cstddef>

namespace WindowsApp::Compute
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

        // Get last error string
        const char* GetLastError() const;

    private:
        CudaContext* m_ctx;
        char m_lastError[512];

        void SetError(const char* msg);
    };
}
