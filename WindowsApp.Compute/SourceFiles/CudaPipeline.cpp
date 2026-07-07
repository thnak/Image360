#include "HeaderFiles/CudaPipeline.h"
#include "HeaderFiles/median_stack.cuh"
#include "HeaderFiles/tensor_ops.cuh"
#include "HeaderFiles/demosaic.cuh"
#include "HeaderFiles/features.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

namespace WindowsApp { namespace Compute
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
    #define CUDA_CHECK(call, lastError) \
        do { \
            cudaError_t err = (call); \
            if (err != cudaSuccess) { \
                snprintf(lastError, sizeof(lastError), \
                    "CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
                return ComputeResult::CUDA_ERROR; \
            } \
        } while (0)

    #define CUDA_CHECK_VOID(call, lastError) \
        do { \
            cudaError_t err = (call); \
            if (err != cudaSuccess) { \
                snprintf(lastError, sizeof(lastError), \
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

        CUDA_CHECK(cudaSetDevice(bestDevice), m_lastError);
        m_ctx->deviceId = bestDevice;

        // Query device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, bestDevice), m_lastError);

        strncpy_s(m_ctx->gpuInfo.name, prop.name, sizeof(m_ctx->gpuInfo.name) - 1);
        m_ctx->gpuInfo.deviceId = bestDevice;
        m_ctx->gpuInfo.totalMemory = prop.totalGlobalMem;
        m_ctx->gpuInfo.computeMajor = prop.major;
        m_ctx->gpuInfo.computeMinor = prop.minor;
        m_ctx->gpuInfo.maxThreadsPerBlock = prop.maxThreadsPerBlock;
        m_ctx->gpuInfo.multiProcessorCount = prop.multiProcessorCount;

        // Tensor cores available on Volta+ (SM 7.0+), Turing (SM 7.5+), Ampere (SM 8.0+), Ada (SM 8.9+), Hopper (SM 9.0+)
        m_ctx->gpuInfo.hasTensorCores = (prop.major >= 7);

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

        CUDA_CHECK(cudaMalloc(&d_src, srcSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_dst, dstSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_invH, 9 * sizeof(float)), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_src, srcData, srcSize, cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_invH, invH, 9 * sizeof(float), cudaMemcpyHostToDevice), m_lastError);

        // Launch kernel
        dim3 block(16, 16);
        dim3 grid((dstW + block.x - 1) / block.x, (dstH + block.y - 1) / block.y);

        WindowsApp::Compute::Kernels::WarpPerspectiveKernel<<<grid, block>>>(
            d_src, d_dst, srcW, srcH, dstW, dstH, d_invH, offsetX, offsetY);

        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        // Copy result back
        CUDA_CHECK(cudaMemcpy(dstData, d_dst, dstSize, cudaMemcpyDeviceToHost), m_lastError);

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

        CUDA_CHECK(cudaMalloc(&d_inputs, totalInputSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_output, singleImageSize), m_lastError);

        // Copy all input images to device
        for (int i = 0; i < numInputs; i++)
        {
            CUDA_CHECK(cudaMemcpy(d_inputs + i * numPixels, inputs[i],
                                   singleImageSize, cudaMemcpyHostToDevice), m_lastError);
        }

        // Launch kernel
        int threadsPerBlock = 256;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

        WindowsApp::Compute::Kernels::MedianStackKernel<<<blocks, threadsPerBlock>>>(
            d_inputs, d_output, numInputs, numPixels, sigmaThreshold);

        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        // Copy result back
        CUDA_CHECK(cudaMemcpy(output, d_output, singleImageSize, cudaMemcpyDeviceToHost), m_lastError);

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

        CUDA_CHECK(cudaMalloc(&d_imgA, imageSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_imgB, imageSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_output, imageSize), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_imgA, imgA, imageSize, cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_imgB, imgB, imageSize, cudaMemcpyHostToDevice), m_lastError);

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

            WindowsApp::Compute::Kernels::Downsample2x<<<grid, block>>>(
                pyramidA[level - 1].d_data, pyramidA[level].d_data,
                srcW, srcH, dstW, dstH);

            WindowsApp::Compute::Kernels::Downsample2x<<<grid, block>>>(
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
            WindowsApp::Compute::Kernels::Upsample2x<<<grid, block>>>(
                pyramidA[level + 1].d_data, d_upsampledA,
                pyramidA[level + 1].w, pyramidA[level + 1].h, w, h);

            WindowsApp::Compute::Kernels::Upsample2x<<<grid, block>>>(
                pyramidB[level + 1].d_data, d_upsampledB,
                pyramidB[level + 1].w, pyramidB[level + 1].h, w, h);

            // Laplacian = current - upsampled
            WindowsApp::Compute::Kernels::LaplacianSubtract<<<grid, block>>>(
                pyramidA[level].d_data, d_upsampledA, pyramidA[level].d_data, w, h);

            WindowsApp::Compute::Kernels::LaplacianSubtract<<<grid, block>>>(
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

            WindowsApp::Compute::Kernels::BlendLaplacianLevels<<<grid, block>>>(
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
            WindowsApp::Compute::Kernels::Upsample2x<<<grid, block>>>(
                blendedPyramid[level + 1].d_data, d_upsampled,
                blendedPyramid[level + 1].w, blendedPyramid[level + 1].h, w, h);

            // Reconstruct: laplacian + upsampled
            WindowsApp::Compute::Kernels::LaplacianReconstruct<<<grid, block>>>(
                blendedPyramid[level].d_data, d_upsampled,
                blendedPyramid[level].d_data, w, h);

            cudaFree(d_upsampled);
        }

        // Copy final result
        CUDA_CHECK(cudaMemcpy(output, blendedPyramid[0].d_data, imageSize, cudaMemcpyDeviceToHost), m_lastError);

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

        CUDA_CHECK(cudaMalloc(&d_data, dataSize), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_data, data, dataSize, cudaMemcpyHostToDevice), m_lastError);

        int threadsPerBlock = 256;
        int blocks = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

        WindowsApp::Compute::Kernels::ApplyGainKernel<<<blocks, threadsPerBlock>>>(d_data, numPixels, gain);

        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        CUDA_CHECK(cudaMemcpy(data, d_data, dataSize, cudaMemcpyDeviceToHost), m_lastError);

        cudaFree(d_data);
        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Kernel 5: GPU Demosaic (RawIngest, docs/ARCHITECTURE.md SS4.1)
    // =========================================================================
    ComputeResult CudaPipeline::DemosaicBayer(
        const unsigned short* cfaData, int width, int height,
        unsigned short blackLevel, const float camMul[4], const float rgbCam[3][4],
        uint32_t filters, unsigned short* rgbOut)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!cfaData || !camMul || !rgbCam || !rgbOut) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        int numPixels = width * height;
        size_t cfaSize = static_cast<size_t>(numPixels) * sizeof(unsigned short);
        size_t rgbSize = static_cast<size_t>(numPixels) * 3 * sizeof(unsigned short);

        unsigned short* d_cfa = nullptr;
        unsigned short* d_rgbA = nullptr;
        unsigned short* d_rgbB = nullptr;
        float* d_rgbCam = nullptr;

        CUDA_CHECK(cudaMalloc(&d_cfa, cfaSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_rgbA, rgbSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_rgbB, rgbSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_rgbCam, 12 * sizeof(float)), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_cfa, cfaData, cfaSize, cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_rgbCam, rgbCam, 12 * sizeof(float), cudaMemcpyHostToDevice), m_lastError);

        int threadsPerBlock = 256;
        int blocks1D = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
        dim3 block2D(16, 16);
        dim3 grid2D((width + block2D.x - 1) / block2D.x, (height + block2D.y - 1) / block2D.y);

        WindowsApp::Compute::Kernels::BlackLevelSubtractKernel<<<blocks1D, threadsPerBlock>>>(
            d_cfa, numPixels, blackLevel);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        WindowsApp::Compute::Kernels::WhiteBalanceKernel<<<grid2D, block2D>>>(
            d_cfa, width, height, camMul[0], camMul[1], camMul[2], camMul[3], filters);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        WindowsApp::Compute::Kernels::DemosaicBayerKernel<<<grid2D, block2D>>>(
            d_cfa, d_rgbA, width, height, filters);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        WindowsApp::Compute::Kernels::ColorMatrixKernel<<<blocks1D, threadsPerBlock>>>(
            d_rgbA, d_rgbB, numPixels, d_rgbCam);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        WindowsApp::Compute::Kernels::ToneCurveKernel<<<blocks1D, threadsPerBlock>>>(
            d_rgbB, d_rgbA, numPixels);
        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        CUDA_CHECK(cudaMemcpy(rgbOut, d_rgbA, rgbSize, cudaMemcpyDeviceToHost), m_lastError);

        cudaFree(d_cfa);
        cudaFree(d_rgbA);
        cudaFree(d_rgbB);
        cudaFree(d_rgbCam);

        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Align: FAST detect + BRIEF describe (docs/ARCHITECTURE.md SS4.2)
    // =========================================================================
    ComputeResult CudaPipeline::DetectAndDescribeFeatures(
        const unsigned char* rgbImage, int width, int height,
        FeaturePoint* outPoints, BriefDescriptor* outDescriptors, int* outCount, int maxPoints)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!rgbImage || !outPoints || !outDescriptors || !outCount) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (width <= 0 || height <= 0 || maxPoints <= 0) { SetError("Invalid dimensions."); return ComputeResult::INVALID_PARAM; }

        constexpr unsigned char kFastThreshold = 20;

        int numPixels = width * height;
        size_t rgbSize = static_cast<size_t>(numPixels) * 3;
        size_t graySize = static_cast<size_t>(numPixels);

        unsigned char* d_rgb = nullptr;
        unsigned char* d_gray = nullptr;
        FeaturePoint* d_points = nullptr;
        BriefDescriptor* d_descriptors = nullptr;
        int* d_count = nullptr;

        CUDA_CHECK(cudaMalloc(&d_rgb, rgbSize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_gray, graySize), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(maxPoints) * sizeof(FeaturePoint)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_descriptors, static_cast<size_t>(maxPoints) * sizeof(BriefDescriptor)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_rgb, rgbImage, rgbSize, cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)), m_lastError);

        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        WindowsApp::Compute::Kernels::RgbToGrayKernel<<<grid, block>>>(d_rgb, d_gray, width, height);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        WindowsApp::Compute::Kernels::FastDetectKernel<<<grid, block>>>(
            d_gray, width, height, d_points, d_count, maxPoints, kFastThreshold);
        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        int detectedCount = 0;
        CUDA_CHECK(cudaMemcpy(&detectedCount, d_count, sizeof(int), cudaMemcpyDeviceToHost), m_lastError);
        int clampedCount = std::min(detectedCount, maxPoints);

        if (clampedCount > 0)
        {
            int threadsPerBlock = 256;
            int blocks = (clampedCount + threadsPerBlock - 1) / threadsPerBlock;
            WindowsApp::Compute::Kernels::BriefDescribeKernel<<<blocks, threadsPerBlock>>>(
                d_gray, width, height, d_points, clampedCount, d_descriptors);
            CUDA_CHECK(cudaGetLastError(), m_lastError);
            CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

            CUDA_CHECK(cudaMemcpy(outPoints, d_points, static_cast<size_t>(clampedCount) * sizeof(FeaturePoint), cudaMemcpyDeviceToHost), m_lastError);
            CUDA_CHECK(cudaMemcpy(outDescriptors, d_descriptors, static_cast<size_t>(clampedCount) * sizeof(BriefDescriptor), cudaMemcpyDeviceToHost), m_lastError);
        }

        *outCount = clampedCount;

        cudaFree(d_rgb);
        cudaFree(d_gray);
        cudaFree(d_points);
        cudaFree(d_descriptors);
        cudaFree(d_count);

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

    // =========================================================================
    // Tensor Core: Batch Matrix Multiply
    // =========================================================================
    ComputeResult CudaPipeline::TensorBatchMatMul(
        const float* A, const float* B, float* C,
        int batchA_M, int batchA_K, int batchB_N,
        int batchSize)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!A || !B || !C) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (!m_ctx->gpuInfo.hasTensorCores) { SetError("Tensor cores not available (requires SM 7.0+)."); return ComputeResult::CUDA_ERROR; }

        // Convert FP32 inputs to FP16
        size_t sizeA = (size_t)batchSize * batchA_M * batchA_K;
        size_t sizeB = (size_t)batchSize * batchA_K * batchB_N;
        size_t sizeC = (size_t)batchSize * batchA_M * batchB_N;

        __half* d_A_half = nullptr;
        __half* d_B_half = nullptr;
        float* d_C = nullptr;
        float* d_A_float = nullptr;
        float* d_B_float = nullptr;

        CUDA_CHECK(cudaMalloc(&d_A_half, sizeA * sizeof(__half)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_B_half, sizeB * sizeof(__half)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_A_float, sizeA * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_B_float, sizeB * sizeof(float)), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_A_float, A, sizeA * sizeof(float), cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_B_float, B, sizeB * sizeof(float), cudaMemcpyHostToDevice), m_lastError);

        // Convert FP32 -> FP16
        int threads = 256;
        WindowsApp::Compute::Kernels::FloatToHalf<<<(sizeA + threads - 1) / threads, threads>>>(d_A_float, d_A_half, (int)sizeA);
        CUDA_CHECK(cudaGetLastError(), m_lastError);
        WindowsApp::Compute::Kernels::FloatToHalf<<<(sizeB + threads - 1) / threads, threads>>>(d_B_float, d_B_half, (int)sizeB);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        // Launch WMMA batched matmul
        // Each block handles one batch, warps handle 16x16 tiles
        int tilesM = (batchA_M + 15) / 16;
        int tilesN = (batchB_N + 15) / 16;
        int warpsNeeded = tilesM * tilesN;
        int threadsPerBlock = min(warpsNeeded * 32, 1024);

        dim3 grid(tilesN, tilesM, batchSize);
        dim3 block(threadsPerBlock);

        WindowsApp::Compute::Kernels::TensorBatchMatMul<<<grid, block>>>(
            d_A_half, d_B_half, d_C,
            batchA_M, batchA_K, batchB_N, batchSize);
        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        CUDA_CHECK(cudaMemcpy(C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost), m_lastError);

        cudaFree(d_A_half);
        cudaFree(d_B_half);
        cudaFree(d_C);
        cudaFree(d_A_float);
        cudaFree(d_B_float);

        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Tensor Core: Homography Estimation
    // =========================================================================
    ComputeResult CudaPipeline::TensorEstimateHomography(
        const float* pointPairs, float* homography, int numPairs)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!pointPairs || !homography) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (numPairs < 4) { SetError("Need at least 4 point correspondences."); return ComputeResult::INVALID_PARAM; }

        // Allocate device memory
        float* d_pairs = nullptr;
        float* d_AtA = nullptr;
        float* d_AtA_copy = nullptr;
        float* d_h = nullptr;

        CUDA_CHECK(cudaMalloc(&d_pairs, numPairs * 4 * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_AtA, 9 * 9 * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_AtA_copy, 9 * 9 * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_h, 9 * sizeof(float)), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_pairs, pointPairs, numPairs * 4 * sizeof(float), cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemset(d_AtA, 0, 9 * 9 * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMemset(d_h, 0, 9 * sizeof(float)), m_lastError);

        // Build AtA
        int threads = 256;
        WindowsApp::Compute::Kernels::TensorBuildAtA<<<1, threads>>>(d_pairs, d_AtA, numPairs);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        // Build Atb from the same point pairs
        // Atb[i] = sum over correspondences of A_row0[i]*xp + A_row1[i]*yp
        // Compute on CPU for simplicity (9 elements)
        float Atb[9] = {};
        for (int p = 0; p < numPairs; p++)
        {
            float x = pointPairs[p * 4 + 0];
            float y = pointPairs[p * 4 + 1];
            float xp = pointPairs[p * 4 + 2];
            float yp = pointPairs[p * 4 + 3];

            float a0[9] = { -x, -y, -1.0f, 0, 0, 0, x * xp, y * xp, xp };
            float a1[9] = { 0, 0, 0, -x, -y, -1.0f, x * yp, y * yp, yp };

            for (int i = 0; i < 9; i++)
                Atb[i] += a0[i] * xp + a1[i] * yp;
        }

        float* d_AtA_ref = nullptr;
        float* d_AtA_out = nullptr;
        CUDA_CHECK(cudaMalloc(&d_AtA_ref, 9 * 9 * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_AtA_out, 9 * 9 * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_AtA_ref, Atb, 9 * sizeof(float), cudaMemcpyHostToDevice), m_lastError);

        // Solve via Cholesky on GPU
        // Copy AtA for in-place decomposition
        CUDA_CHECK(cudaMemcpy(d_AtA_copy, d_AtA, 9 * 9 * sizeof(float), cudaMemcpyDeviceToDevice), m_lastError);

        WindowsApp::Compute::Kernels::TensorSolveHomography<<<1, 1>>>(d_AtA_copy, d_AtA_ref, d_h);
        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        // Copy result
        CUDA_CHECK(cudaMemcpy(homography, d_h, 9 * sizeof(float), cudaMemcpyDeviceToHost), m_lastError);

        // Normalize so h[8] = 1
        if (fabsf(homography[8]) > 1e-10f)
        {
            float inv = 1.0f / homography[8];
            for (int i = 0; i < 9; i++)
                homography[i] *= inv;
        }

        cudaFree(d_pairs);
        cudaFree(d_AtA);
        cudaFree(d_AtA_copy);
        cudaFree(d_h);
        cudaFree(d_AtA_ref);
        cudaFree(d_AtA_out);

        return ComputeResult::SUCCESS;
    }

    // =========================================================================
    // Tensor Core: Normal Equations for Bundle Adjustment
    // =========================================================================
    ComputeResult CudaPipeline::TensorSolveNormalEquations(
        const float* J, const float* r, float* delta,
        int numResiduals, int numParams, float lambda)
    {
        if (!m_ctx->initialized) { SetError("Not initialized."); return ComputeResult::CUDA_ERROR; }
        if (!J || !r || !delta) { SetError("Null pointer argument."); return ComputeResult::INVALID_PARAM; }
        if (!m_ctx->gpuInfo.hasTensorCores) { SetError("Tensor cores not available (requires SM 7.0+)."); return ComputeResult::CUDA_ERROR; }

        size_t J_size = (size_t)numResiduals * numParams;
        size_t JtJ_size = (size_t)numParams * numParams;

        // Allocate device memory
        __half* d_J_half = nullptr;
        float* d_J = nullptr;
        float* d_r = nullptr;
        float* d_JtJ = nullptr;
        float* d_Jtr = nullptr;

        CUDA_CHECK(cudaMalloc(&d_J_half, J_size * sizeof(__half)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_J, J_size * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_r, numResiduals * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_JtJ, JtJ_size * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMalloc(&d_Jtr, numParams * sizeof(float)), m_lastError);

        CUDA_CHECK(cudaMemcpy(d_J, J, J_size * sizeof(float), cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemcpy(d_r, r, numResiduals * sizeof(float), cudaMemcpyHostToDevice), m_lastError);
        CUDA_CHECK(cudaMemset(d_JtJ, 0, JtJ_size * sizeof(float)), m_lastError);
        CUDA_CHECK(cudaMemset(d_Jtr, 0, numParams * sizeof(float)), m_lastError);

        // Convert J to FP16 for tensor core matmul
        int threads = 256;
        WindowsApp::Compute::Kernels::FloatToHalf<<<(J_size + threads - 1) / threads, threads>>>(d_J, d_J_half, (int)J_size);
        CUDA_CHECK(cudaGetLastError(), m_lastError);

        // Compute normal equations: JtJ = J^T * J + lambda * I, Jtr = J^T * r
        int totalElements = numParams * numParams;
        WindowsApp::Compute::Kernels::TensorNormalEquations<<<(totalElements + threads - 1) / threads, threads>>>(
            d_J_half, d_r, d_JtJ, d_Jtr, numResiduals, numParams);
        CUDA_CHECK(cudaGetLastError(), m_lastError);
        CUDA_CHECK(cudaDeviceSynchronize(), m_lastError);

        // Add LM damping: JtJ += lambda * diag(JtJ)
        // Done on CPU for small systems, or on GPU for large ones
        std::vector<float> h_JtJ(JtJ_size);
        std::vector<float> h_Jtr(numParams);
        CUDA_CHECK(cudaMemcpy(h_JtJ.data(), d_JtJ, JtJ_size * sizeof(float), cudaMemcpyDeviceToHost), m_lastError);
        CUDA_CHECK(cudaMemcpy(h_Jtr.data(), d_Jtr, numParams * sizeof(float), cudaMemcpyDeviceToHost), m_lastError);

        for (int i = 0; i < numParams; i++)
        {
            h_JtJ[i * numParams + i] *= (1.0f + lambda);
        }

        // Solve using Cholesky (CPU for small systems typical in BA)
        // For production, use cuBLAS/cuSolver on GPU
        std::vector<float> L(numParams * numParams, 0.0f);
        std::vector<float> y(numParams, 0.0f);

        // Cholesky decomposition
        for (int i = 0; i < numParams; i++)
        {
            for (int j = 0; j <= i; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < j; k++)
                    sum += L[i * numParams + k] * L[j * numParams + k];

                if (i == j)
                {
                    float diag = h_JtJ[i * numParams + i] - sum;
                    L[i * numParams + j] = sqrtf(fmaxf(diag, 1e-10f));
                }
                else
                {
                    L[i * numParams + j] = (h_JtJ[i * numParams + j] - sum) / L[j * numParams + j];
                }
            }
        }

        // Forward substitution
        for (int i = 0; i < numParams; i++)
        {
            float sum = 0.0f;
            for (int k = 0; k < i; k++)
                sum += L[i * numParams + k] * y[k];
            y[i] = (h_Jtr[i] - sum) / L[i * numParams + i];
        }

        // Backward substitution
        for (int i = numParams - 1; i >= 0; i--)
        {
            float sum = 0.0f;
            for (int k = i + 1; k < numParams; k++)
                sum += L[k * numParams + i] * delta[k];
            delta[i] = (y[i] - sum) / L[i * numParams + i];
        }

        cudaFree(d_J_half);
        cudaFree(d_J);
        cudaFree(d_r);
        cudaFree(d_JtJ);
        cudaFree(d_Jtr);

        return ComputeResult::SUCCESS;
    }
    }
}
