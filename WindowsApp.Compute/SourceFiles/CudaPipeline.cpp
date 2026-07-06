#include "HeaderFiles/CudaPipeline.h"
#include "HeaderFiles/median_stack.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace WindowsApp::Compute
{
    struct CudaContext
    {
        int deviceId = -1;
        GpuInfo gpuInfo;
        bool initialized = false;

        // Gaussian kernel for multi-band blending
        float* d_gaussKernel = nullptr;
        int gaussKernelRadius = 0;

        // Helper: compute 1D Gaussian kernel
        std::vector<float> ComputeGaussianKernel(int radius, float sigma)
        {
            int size = 2 * radius + 1;
            std::vector<float> kernel(size);
            float sum = 0.0f;

            for (int i = 0; i < size; i++)
            {
                float x = static_cast<float>(i - radius);
                kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
                sum += kernel[i];
            }

            // Normalize
            for (auto& v : kernel) v /= sum;
            return kernel;
        }
    };

    // CUDA error checking macro
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = (call); \
            if (err != cudaSuccess) { \
                snprintf(m_lastError, sizeof(m_lastError), \
                    "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
                return ComputeResult::CUDA_ERROR; \
            } \
        } while (0)

    #define CUDA_CHECK_VOID(call) \
        do { \
            cudaError_t err = (call); \
            if (err != cudaSuccess) { \
                snprintf(m_lastError, sizeof(m_lastError), \
                    "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
                return; \
            } \
        } while (0)

    CudaPipeline::CudaPipeline()
        : m_ctx(new CudaContext())
        , m_lastError{}
    {
    }

    CudaPipeline::~CudaPipeline()
    {
        Shutdown();
        delete m_ctx;
    }

    ComputeResult CudaPipeline::Initialize()
    {
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0)
        {
            SetError("No CUDA-capable GPU found.");
            return ComputeResult::NO_GPU;
        }

        // Select device with most memory
        int bestDevice = 0;
        size_t maxMemory = 0;

        for (int i = 0; i < deviceCount; i++)
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);

            if (prop.totalGlobalMem > maxMemory)
            {
                maxMemory = prop.totalGlobalMem;
                bestDevice = i;
            }
        }

        CUDA_CHECK(cudaSetDevice(bestDevice));
        m_ctx->deviceId = bestDevice;

        // Query device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, bestDevice));

        strncpy_s(m_ctx->gpuInfo.name, prop.name, sizeof(m_ctx->gpuInfo.name) - 1);
        m_ctx->gpuInfo.deviceId = bestDevice;
        m_ctx->gpuInfo.totalMemory = prop.totalGlobalMem;
        m_ctx->gpuInfo.computeMajor = prop.major;
        m_ctx->gpuInfo.computeMinor = prop.minor;
        m_ctx->gpuInfo.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        m_ctx->gpuInfo.multiProcessorCount = prop.multiProcessorCount;

        // Query free memory
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        m_ctx->gpuInfo.freeMemory = free;

        m_ctx->initialized = true;
        return ComputeResult::SUCCESS;
    }

    void CudaPipeline::Shutdown()
    {
        if (m_ctx->initialized)
        {
            if (m_ctx->d_gaussKernel)
            {
                cudaFree(m_ctx->d_gaussKernel);
                m_ctx->d_gaussKernel = nullptr;
            }
            cudaDeviceReset();
            m_ctx->initialized = false;
        }
    }

    bool CudaPipeline::IsInitialized() const
    {
        return m_ctx->initialized;
    }

    GpuInfo CudaPipeline::GetGpuInfo() const
    {
        return m_ctx->gpuInfo;
    }

    // =========================================================================
    // Kernel 1: Perspective Warp
    // =========================================================================
    ComputeResult CudaPipeline::WarpPerspective(
        const unsigned short* srcData, int srcW, int srcH,
        unsigned short* dstData, int dstW, int dstH,
        const float* homography, int offsetX, int offsetY)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!srcData || !dstData || !homography) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (srcW <= 0 || srcH <= 0 || dstW <= 0 || dstH <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        // Compute inverse homography on CPU
        const float* m = homography;
        float det = m[0] * (m[4] * m[8] - m[5] * m[7])
                  - m[1] * (m[3] * m[8] - m[5] * m[6])
                  + m[2] * (m[3] * m[7] - m[4] * m[6]);

        if (fabsf(det) < 1e-10f) { SetError("Singular homography matrix."); return ComputeResult::INVALID_PARAM; }

        float invDet = 1.0f / det;
        float invH[9] = {
            (m[4] * m[8] - m[5] * m[7]) * invDet,
            (m[2] * m[7] - m[1] * m[8]) * invDet,
            (m[1] * m[5] - m[2] * m[4]) * invDet,
            (m[5] * m[6] - m[3] * m[8]) * invDet,
            (m[0] * m[8] - m[2] * m[6]) * invDet,
            (m[2] * m[3] - m[0] * m[5]) * invDet,
            (m[3] * m[7] - m[4] * m[6]) * invDet,
            (m[1] * m[6] - m[0] * m[7]) * invDet,
            (m[0] * m[4] - m[1] * m[3]) * invDet
        };

        // Allocate device memory
        size_t srcSize = srcW * srcH * 3 * sizeof(unsigned short);
        size_t dstSize = dstW * dstH * 3 * sizeof(unsigned short);

        unsigned short* d_src = nullptr;
        unsigned short* d_dst = nullptr;
        float* d_invH = nullptr;

        CUDA_CHECK(cudaMalloc(&d_src, srcSize));
        CUDA_CHECK(cudaMalloc(&d_dst, dstSize));
        CUDA_CHECK(cudaMalloc(&d_invH, 9 * sizeof(float)));

        CUDA_CHECK(cudaMemcpy(d_src, srcData, srcSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_invH, invH, 9 * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel
        dim3 block(16, 16);
        dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

        Kernels::WarpPerspectiveKernel<<<grid, block>>>(
            d_src, d_dst, srcW, srcH, dstW, dstH, d_invH, offsetX, offsetY);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK(cudaMemcpy(dstData, d_dst, dstSize, cudaMemcpyDeviceToHost));

        // Cleanup
        cudaFree(d_src);
        cudaFree(d_dst);
        cudaFree(d_invH);

        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Kernel 2: Median Stack
    // =========================================================================
    ComputeResult CudaPipeline::MedianStack(
        const unsigned short** inputs, int numInputs,
        unsigned short* output, int width, int height,
        float sigmaThreshold)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!inputs || !output) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numInputs <= 0 || numInputs > 32) { SetError("numInputs must be 1-32."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        int numPixels = width * height * 3;
        size_t singleImageSize = numPixels * sizeof(unsigned short);
        size_t totalInputSize = numInputs * singleImageSize;

        // Allocate device memory for all inputs (flattened)
        unsigned short* d_inputs = nullptr;
        unsigned short* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_inputs, totalInputSize));
        CUDA_CHECK(cudaMalloc(&d_output, singleImageSize));

        // Copy all input images to device
        for (int i = 0; i < numInputs; i++)
        {
            CUDA_CHECK(cudaMemcpy(d_inputs + i * numPixels, inputs[i],
                                   singleImageSize, cudaMemcpyHostToDevice));
        }

        // Launch kernel
        int threadsPerBlock = 256;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

        Kernels::MedianStackKernel<<<blocks, threadsPerBlock>>>(
            d_inputs, d_output, numInputs, numPixels, sigmaThreshold);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy result back
        CUDA_CHECK(cudaMemcpy(output, d_output, singleImageSize, cudaMemcpyDeviceToHost));

        // Cleanup
        cudaFree(d_inputs);
        cudaFree(d_output);

        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Kernel 3: Multi-Band Blending (Laplacian Pyramid)
    // =========================================================================
    ComputeResult CudaPipeline::MultiBandBlend(
        const unsigned short* imgA, const unsigned short* imgB,
        unsigned short* output, int width, int height,
        int numBands)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!imgA || !imgB || !output) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }
        if (numBands < 1 || numBands > 10) { SetError("numBands must be 1-10."); return ComputeResult::INVALID_PARAM; }

        dim3 block(16, 16);

        // Allocate device memory for input images
        size_t imageSize = width * height * 3 * sizeof(unsigned short);

        unsigned short* d_imgA = nullptr;
        unsigned short* d_imgB = nullptr;
        unsigned short* d_output = nullptr;

        CUDA_CHECK(cudaMalloc(&d_imgA, imageSize));
        CUDA_CHECK(cudaMalloc(&d_imgB, imageSize));
        CUDA_CHECK(cudaMalloc(&d_output, imageSize));

        CUDA_CHECK(cudaMemcpy(d_imgA, imgA, imageSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_imgB, imgB, imageSize, cudaMemcpyHostToDevice));

        // Build Laplacian pyramids for both images
        // For simplicity, process each color channel separately
        // In production, this would be optimized to handle all 3 channels together

        struct PyramidLevel
        {
            unsigned short* d_data;
            int w, h;
        };

        // Allocate pyramid levels for image A and B
        std::vector<PyramidLevel> pyramidA(numBands);
        std::vector<PyramidLevel> pyramidB(numBands);
        std::vector<PyramidLevel> blendedPyramid(numBands);

        int curW = width;
        int curH = height;

        // Allocate all levels
        for (int level = 0; level < numBands; level++)
        {
            size_t levelSize = curW * curH * 3 * sizeof(unsigned short);
            cudaMalloc(&pyramidA[level].d_data, levelSize);
            cudaMalloc(&pyramidB[level].d_data, levelSize);
            cudaMalloc(&blendedPyramid[level].d_data, levelSize);
            pyramidA[level].w = curW;
            pyramidA[level].h = curH;
            pyramidB[level].w = curW;
            pyramidB[level].h = curH;
            blendedPyramid[level].w = curW;
            blendedPyramid[level].h = curH;

            curW = (curW + 1) / 2;
            curH = (curH + 1) / 2;
        }

        // Build Gaussian pyramid and extract Laplacian levels
        // Level 0 = original image
        // Level i = downsampled from level i-1
        // Laplacian[i] = Gaussian[i] - Upsample(Gaussian[i+1])
        // Top level Laplacian = Gaussian[top] (no subtraction)

        // Copy images to level 0
        cudaMemcpy(pyramidA[0].d_data, imgA, imageSize, cudaMemcpyHostToDevice);
        cudaMemcpy(pyramidB[0].d_data, imgB, imageSize, cudaMemcpyHostToDevice);

        // Build Gaussian pyramid by downsampling
        for (int level = 1; level < numBands; level++)
        {
            int srcW = pyramidA[level - 1].w;
            int srcH = pyramidA[level - 1].h;
            int dstW = pyramidA[level].w;
            int dstH = pyramidA[level].h;

            dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

            Kernels::Downsample2x<<<grid, block>>>(
                pyramidA[level - 1].d_data, pyramidA[level].d_data,
                srcW, srcH, dstW, dstH);

            Kernels::Downsample2x<<<grid, block>>>(
                pyramidB[level - 1].d_data, pyramidB[level].d_data,
                srcW, srcH, dstW, dstH);
        }

        // Extract Laplacian levels
        // Top level is just the Gaussian (low-frequency)
        // Lower levels = Gaussian[level] - Upsample(Gaussian[level+1])
        for (int level = 0; level < numBands - 1; level++)
        {
            int w = pyramidA[level].w;
            int h = pyramidA[level].h;

            // Allocate upsampled buffer
            unsigned short* d_upsampledA = nullptr;
            unsigned short* d_upsampledB = nullptr;
            size_t upSize = w * h * 3 * sizeof(unsigned short);
            cudaMalloc(&d_upsampledA, upSize);
            cudaMalloc(&d_upsampledB, upSize);

            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

            // Upsample lower level to current size
            Kernels::Upsample2x<<<grid, block>>>(
                pyramidA[level + 1].d_data, d_upsampledA,
                pyramidA[level + 1].w, pyramidA[level + 1].h, w, h);

            Kernels::Upsample2x<<<grid, block>>>(
                pyramidB[level + 1].d_data, d_upsampledB,
                pyramidB[level + 1].w, pyramidB[level + 1].h, w, h);

            // Laplacian = current - upsampled
            Kernels::LaplacianSubtract<<<grid, block>>>(
                pyramidA[level].d_data, d_upsampledA, pyramidA[level].d_data, w, h);

            Kernels::LaplacianSubtract<<<grid, block>>>(
                pyramidB[level].d_data, d_upsampledB, pyramidB[level].d_data, w, h);

            cudaFree(d_upsampledA);
            cudaFree(d_upsampledB);
        }

        // Blend each Laplacian level (equal weight 50/50)
        for (int level = 0; level < numBands; level++)
        {
            int w = blendedPyramid[level].w;
            int h = blendedPyramid[level].h;
            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

            float weightA = 0.5f;
            float weightB = 0.5f;

            // Top level: simple average (low frequency, smooth blend)
            // Lower levels: could use different weights based on edge masks
            if (level == numBands - 1)
            {
                weightA = 0.5f;
                weightB = 0.5f;
            }

            Kernels::BlendLaplacianLevels<<<grid, block>>>(
                pyramidA[level].d_data, pyramidB[level].d_data,
                blendedPyramid[level].d_data, w, h, weightA, weightB);
        }

        // Reconstruct from blended Laplacian pyramid
        // Start from top, add to level below, repeat
        for (int level = numBands - 2; level >= 0; level--)
        {
            int w = pyramidA[level].w;
            int h = pyramidA[level].h;

            unsigned short* d_upsampled = nullptr;
            size_t upSize = w * h * 3 * sizeof(unsigned short);
            cudaMalloc(&d_upsampled, upSize);

            dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

            // Upsample blended lower level
            Kernels::Upsample2x<<<grid, block>>>(
                blendedPyramid[level + 1].d_data, d_upsampled,
                blendedPyramid[level + 1].w, blendedPyramid[level + 1].h, w, h);

            // Reconstruct: laplacian + upsampled
            Kernels::LaplacianReconstruct<<<grid, block>>>(
                blendedPyramid[level].d_data, d_upsampled,
                blendedPyramid[level].d_data, w, h);

            cudaFree(d_upsampled);
        }

        // Copy final result
        CUDA_CHECK(cudaMemcpy(output, blendedPyramid[0].d_data, imageSize, cudaMemcpyDeviceToHost));

        // Cleanup
        for (int level = 0; level < numBands; level++)
        {
            cudaFree(pyramidA[level].d_data);
            cudaFree(pyramidB[level].d_data);
            cudaFree(blendedPyramid[level].d_data);
        }
        cudaFree(d_imgA);
        cudaFree(d_imgB);
        cudaFree(d_output);

        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Kernel 4: Gain Compensation
    // =========================================================================
    ComputeResult CudaPipeline::ApplyGain(
        unsigned short* data, int numPixels, float gain)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!data) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numPixels <= 0) { SetError("Invalid pixel count."); return ComputeResult::INVALID_PARAM; }
        if (fabsf(gain - 1.0f) < 1e-6f) return ComputeResult::SUCCESS; // No-op for gain ~= 1.0

        size_t dataSize = numPixels * sizeof(unsigned short);
        unsigned short* d_data = nullptr;

        CUDA_CHECK(cudaMalloc(&d_data, dataSize));
        CUDA_CHECK(cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

        Kernels::ApplyGainKernel<<<blocks, threadsPerBlock>>>(d_data, numPixels, gain);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(data, d_data, dataSize, cudaMemcpyDeviceToHost));

        cudaFree(d_data);
        return ComputeResult::SUCCESS;
    }

    const char* CudaPipeline::GetLastError() const
    {
        return m_lastError;
    }

    void CudaPipeline::SetError(const char* msg)
    {
        strncpy_s(m_lastError, sizeof(m_lastError), msg, _TRUNCATE);
    }
}
