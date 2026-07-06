#include "pch.h"
#include "HeaderFiles/WorkflowController.h"
#include "HeaderFiles/ImageLoader.h"
#include "HeaderFiles/CudaPipeline.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace WindowsApp::Core
{
    void WorkflowController::Initialize(ProgressCallback onProgress, LogCallback onLog, StageCallback onStage)
    {
        m_onProgress = std::move(onProgress);
        m_onLog = std::move(onLog);
        m_onStage = std::move(onStage);
    }

    bool WorkflowController::StartProcessing(std::stop_token stopToken, const std::wstring& projectPath)
    {
        Log(L"Loading project: " + projectPath);

        if (!m_projectManager.LoadProject(projectPath))
        {
            Log(L"ERROR: Failed to load project.");
            SetStage(PipelineStage::FAILED);
            return false;
        }

        Log(L"Project loaded. Images: " + std::to_wstring(m_projectManager.GetInputImages().size())
            + L", Chunks: " + std::to_wstring(m_projectManager.GetChunks().size()));

        // Initialize CUDA pipeline
        m_cudaPipeline = std::make_unique<WindowsApp::Compute::CudaPipeline>();
        auto gpuResult = m_cudaPipeline->Initialize();
        if (gpuResult == WindowsApp::Compute::ComputeResult::SUCCESS)
        {
            auto gpuInfo = m_cudaPipeline->GetGpuInfo();
            Log(L"GPU initialized: " + std::wstring(gpuInfo.name, gpuInfo.name + strlen(gpuInfo.name)));
            Log(L"  VRAM: " + std::to_wstring(gpuInfo.totalMemory / (1024 * 1024)) + L" MB");
            m_gpuAvailable = true;
        }
        else
        {
            Log(L"WARNING: GPU not available (" + std::wstring(m_cudaPipeline->GetLastError(),
                m_cudaPipeline->GetLastError() + strlen(m_cudaPipeline->GetLastError())) + L"), falling back to CPU.");
            m_gpuAvailable = false;
        }

        // Stage 1: Alignment
        if (!Stage1_Align(stopToken))
        {
            if (stopToken.stop_requested())
            {
                SetStage(PipelineStage::CANCELLED);
                Log(L"Cancelled during Stage 1.");
            }
            else
            {
                SetStage(PipelineStage::FAILED);
                Log(L"ERROR: Stage 1 (Alignment) failed.");
            }
            return false;
        }

        if (stopToken.stop_requested()) { SetStage(PipelineStage::CANCELLED); return false; }

        // Stage 2: Optimization
        if (!Stage2_Optimize(stopToken))
        {
            if (stopToken.stop_requested())
            {
                SetStage(PipelineStage::CANCELLED);
                Log(L"Cancelled during Stage 2.");
            }
            else
            {
                SetStage(PipelineStage::FAILED);
                Log(L"ERROR: Stage 2 (Optimization) failed.");
            }
            return false;
        }

        if (stopToken.stop_requested()) { SetStage(PipelineStage::CANCELLED); return false; }

        // Stage 3: Rendering
        if (!Stage3_Render(stopToken))
        {
            if (stopToken.stop_requested())
            {
                SetStage(PipelineStage::CANCELLED);
                Log(L"Cancelled during Stage 3.");
            }
            else
            {
                SetStage(PipelineStage::FAILED);
                Log(L"ERROR: Stage 3 (Rendering) failed.");
            }
            return false;
        }

        SetStage(PipelineStage::COMPLETED);
        m_overallProgress = 1.0f;
        Log(L"Pipeline completed successfully.");
        return true;
    }

    void WorkflowController::Cancel()
    {
        Log(L"Cancel requested.");
    }

    // =========================================================================
    // Stage 1: Low-Res Alignment (Explore & Position)
    // =========================================================================
    bool WorkflowController::Stage1_Align(std::stop_token& stopToken)
    {
        SetStage(PipelineStage::STAGE1_ALIGN);
        Log(L"--- Stage 1: Low-Res Alignment ---");

        const auto& images = m_projectManager.GetInputImages();
        if (images.size() < 2)
        {
            Log(L"Need at least 2 images for alignment.");
            return false;
        }

        // Load low-res proxies (10% resolution)
        std::vector<PixelBuffer> lowResImages(images.size());
        for (size_t i = 0; i < images.size(); i++)
        {
            if (stopToken.stop_requested()) return false;

            Log(L"Loading low-res proxy: " + images[i].file_path);
            if (!LoadImageLowRes(images[i].file_path, lowResImages[i], 10))
            {
                Log(L"WARNING: Failed to load " + images[i].file_path + L", skipping.");
                continue;
            }

            UpdateProgress("", static_cast<float>(i) / images.size() * 0.15f);
        }

        // Extract features and match adjacent pairs
        Log(L"Extracting features...");
        std::vector<std::vector<std::array<float, 2>>> allKeypoints(images.size());

        for (size_t i = 0; i < images.size(); i++)
        {
            if (stopToken.stop_requested()) return false;
            ExtractFeatures(lowResImages[i], allKeypoints[i]);
            Log(L"Image " + std::to_wstring(i) + L": " + std::to_wstring(allKeypoints[i].size()) + L" keypoints");
        }

        // Match and compute homographies for adjacent pairs
        Log(L"Matching features and computing homographies...");
        for (size_t i = 0; i < images.size() - 1; i++)
        {
            if (stopToken.stop_requested()) return false;

            std::vector<std::pair<int, int>> matches;
            MatchFeatures(allKeypoints[i], allKeypoints[i + 1], matches);

            if (matches.size() < 4)
            {
                Log(L"WARNING: Insufficient matches between image " + std::to_wstring(i)
                    + L" and " + std::to_wstring(i + 1));
                continue;
            }

            // Build point arrays from matches
            std::vector<std::array<float, 2>> pts1, pts2;
            for (const auto& m : matches)
            {
                pts1.push_back(allKeypoints[i][m.first]);
                pts2.push_back(allKeypoints[i + 1][m.second]);
            }

            Homography h;
            if (ComputeHomographyRANSAC(pts1, pts2, h))
            {
                m_projectManager.AddInputImage(images[i + 1].file_path, h);
                Log(L"Homography computed for pair " + std::to_wstring(i) + L"-" + std::to_wstring(i + 1));
            }

            UpdateProgress("", 0.10f + static_cast<float>(i) / images.size() * 0.05f);
        }

        // Free low-res buffers
        lowResImages.clear();

        Log(L"Stage 1 complete.");
        UpdateProgress("", 0.15f);
        return true;
    }

    bool WorkflowController::LoadImageLowRes(const std::wstring& path, PixelBuffer& output, int scalePercent)
    {
        ImageLoader loader;

        if (!loader.Open(path))
        {
            Log(L"Failed to open image: " + path + L" - " + loader.GetLastError());
            return false;
        }

        // Get metadata for logging
        ImageMetadata meta;
        if (loader.GetMetadata(meta))
        {
            Log(L"  Camera: " + meta.cameraMake + L" " + meta.cameraModel);
            Log(L"  Size: " + std::to_wstring(meta.width) + L"x" + std::to_wstring(meta.height));
        }

        // Decode at reduced resolution
        if (!loader.DecodeThumbnail(output, scalePercent))
        {
            Log(L"Failed to decode thumbnail: " + path + L" - " + loader.GetLastError());
            return false;
        }

        Log(L"  Decoded: " + std::to_wstring(output.width) + L"x" + std::to_wstring(output.height)
            + L" (" + std::to_wstring(output.data.size()) + L" pixels)");

        return true;
    }

    bool WorkflowController::ExtractFeatures(const PixelBuffer& image, std::vector<std::array<float, 2>>& keypoints)
    {
        // TODO: Implement ORB or SIFT feature extraction
        // For now, generate placeholder keypoints on a grid
        keypoints.clear();

        if (image.width == 0 || image.height == 0) return true;

        const int step = 32;
        for (int y = step; y < image.height - step; y += step)
        {
            for (int x = step; x < image.width - step; x += step)
            {
                keypoints.push_back({ static_cast<float>(x), static_cast<float>(y) });
            }
        }
        return true;
    }

    bool WorkflowController::MatchFeatures(const std::vector<std::array<float, 2>>& kp1,
                                            const std::vector<std::array<float, 2>>& kp2,
                                            std::vector<std::pair<int, int>>& matches)
    {
        // TODO: Implement brute-force or FLANN matching with ratio test
        // Placeholder: match nearest neighbors within a threshold
        matches.clear();

        const float maxDist = 50.0f;
        for (size_t i = 0; i < kp1.size(); i++)
        {
            float bestDist = maxDist;
            int bestIdx = -1;

            for (size_t j = 0; j < kp2.size(); j++)
            {
                float dx = kp1[i][0] - kp2[j][0];
                float dy = kp1[i][1] - kp2[j][1];
                float dist = std::sqrt(dx * dx + dy * dy);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = static_cast<int>(j);
                }
            }

            if (bestIdx >= 0)
            {
                matches.push_back({ static_cast<int>(i), bestIdx });
            }
        }
        return true;
    }

    bool WorkflowController::ComputeHomographyRANSAC(const std::vector<std::array<float, 2>>& pts1,
                                                      const std::vector<std::array<float, 2>>& pts2,
                                                      Homography& outH)
    {
        // TODO: Implement proper RANSAC with DLT (Direct Linear Transform)
        // Placeholder: identity homography
        if (pts1.size() < 4) return false;

        // Simple least-squares approximation for translation (not full homography)
        float dx = 0, dy = 0;
        for (size_t i = 0; i < pts1.size(); i++)
        {
            dx += pts2[i][0] - pts1[i][0];
            dy += pts2[i][1] - pts1[i][1];
        }
        dx /= pts1.size();
        dy /= pts1.size();

        outH.h = { 1.0f, 0.0f, dx, 0.0f, 1.0f, dy, 0.0f, 0.0f, 1.0f };
        return true;
    }

    // =========================================================================
    // Stage 2: Global Optimization & Calibration
    // =========================================================================
    bool WorkflowController::Stage2_Optimize(std::stop_token& stopToken)
    {
        SetStage(PipelineStage::STAGE2_OPTIMIZE);
        Log(L"--- Stage 2: Global Optimization ---");

        if (stopToken.stop_requested()) return false;

        Log(L"Computing gain compensation...");
        if (!ComputeGainCompensation())
        {
            Log(L"WARNING: Gain compensation failed, using defaults.");
        }
        UpdateProgress("", 0.20f);

        if (stopToken.stop_requested()) return false;

        Log(L"Applying color transfer...");
        if (!ApplyColorTransfer())
        {
            Log(L"WARNING: Color transfer failed, skipping.");
        }
        UpdateProgress("", 0.25f);

        if (stopToken.stop_requested()) return false;

        Log(L"Running bundle adjustment...");
        if (!RunBundleAdjustment())
        {
            Log(L"WARNING: Bundle adjustment failed, using current homographies.");
        }
        UpdateProgress("", 0.30f);

        Log(L"Stage 2 complete.");
        return true;
    }

    bool WorkflowController::ComputeGainCompensation()
    {
        // TODO: Solve least-squares system for gain factors g_i
        // Minimize sum of (g_i * I_i - g_j * I_j)^2 over overlapping regions
        // For now, set all gains to 1.0
        const auto& images = m_projectManager.GetInputImages();
        for (const auto& img : images)
        {
            m_projectManager.UpdateImageGain(img.id, 1.0f);
        }
        return true;
    }

    bool WorkflowController::ApplyColorTransfer()
    {
        // TODO: Implement color transfer using Reinhard's method
        // Convert to LAB color space, match mean and std dev across images
        return true;
    }

    bool WorkflowController::RunBundleAdjustment()
    {
        // TODO: Implement sparse bundle adjustment (Ceres Solver or custom LM)
        // Optimize all homographies simultaneously to minimize reprojection error
        return true;
    }

    // =========================================================================
    // Stage 3: High-Res Tile Rendering
    // =========================================================================
    bool WorkflowController::Stage3_Render(std::stop_token& stopToken)
    {
        SetStage(PipelineStage::STAGE3_RENDER);
        Log(L"--- Stage 3: High-Res Tile Rendering ---");

        auto chunks = m_projectManager.GetChunks();
        int totalChunks = static_cast<int>(chunks.size());
        int completedChunks = 0;

        for (auto& chunk : chunks)
        {
            if (stopToken.stop_requested()) return false;

            // Skip already completed chunks (resume support)
            if (chunk.status == ChunkStatus::COMPLETED)
            {
                completedChunks++;
                continue;
            }

            Log(L"Processing chunk: " + std::wstring(chunk.id.begin(), chunk.id.end()));
            m_projectManager.UpdateChunkStatus(chunk.id, ChunkStatus::PROCESSING, L"");

            if (ProcessChunk(chunk, stopToken))
            {
                m_projectManager.UpdateChunkStatus(chunk.id, ChunkStatus::COMPLETED, chunk.cache_path);
                completedChunks++;
            }
            else
            {
                m_projectManager.UpdateChunkStatus(chunk.id, ChunkStatus::FAILED, L"");
                Log(L"WARNING: Chunk " + std::wstring(chunk.id.begin(), chunk.id.end()) + L" failed.");
            }

            float stage3Progress = static_cast<float>(completedChunks) / totalChunks;
            UpdateProgress(chunk.id, 0.30f + stage3Progress * 0.70f);
        }

        Log(L"Stage 3 complete. " + std::to_wstring(completedChunks) + L"/" + std::to_wstring(totalChunks) + L" chunks rendered.");
        return true;
    }

    bool WorkflowController::ProcessChunk(ChunkModel& chunk, std::stop_token& stopToken)
    {
        if (stopToken.stop_requested()) return false;

        const auto& images = m_projectManager.GetInputImages();

        // 1. Determine which images overlap this chunk
        std::vector<size_t> overlappingImages;
        for (size_t i = 0; i < images.size(); i++)
        {
            // TODO: Check if image footprint intersects chunk bounds using homography
            overlappingImages.push_back(i);
        }

        if (overlappingImages.empty())
        {
            Log(L"No images overlap chunk " + std::wstring(chunk.id.begin(), chunk.id.end()));
            return true; // Empty chunk, not a failure
        }

        // 2. For each overlapping image, load ROI and warp
        std::vector<PixelBuffer> warpedImages(overlappingImages.size());
        for (size_t i = 0; i < overlappingImages.size(); i++)
        {
            if (stopToken.stop_requested()) return false;

            size_t imgIdx = overlappingImages[i];
            const auto& img = images[imgIdx];

            // Load ROI from the full-res image using LibRaw
            ImageLoader loader;
            PixelBuffer fullRes;

            if (!loader.Open(img.file_path))
            {
                Log(L"Failed to open image " + std::to_wstring(imgIdx) + L": " + loader.GetLastError());
                continue;
            }

            if (!loader.DecodeROI(chunk.x_offset, chunk.y_offset, chunk.width, chunk.height, fullRes))
            {
                Log(L"Failed to decode ROI from image " + std::to_wstring(imgIdx) + L": " + loader.GetLastError());
                continue;
            }

            // Apply gain compensation
            float gain = img.gain;
            if (gain != 1.0f)
            {
                if (m_gpuAvailable && m_cudaPipeline)
                {
                    m_cudaPipeline->ApplyGain(fullRes.data.data(),
                        static_cast<int>(fullRes.data.size()), gain);
                }
                else
                {
                    for (auto& pixel : fullRes.data)
                    {
                        pixel = static_cast<unsigned short>(
                            std::clamp(static_cast<float>(pixel) * gain, 0.0f, 65535.0f));
                    }
                }
            }

            // Warp perspective using homography
            if (!WarpPerspective(fullRes, warpedImages[i], img.homography,
                                 chunk.x_offset, chunk.y_offset, chunk.width, chunk.height))
            {
                Log(L"Warp failed for image " + std::to_wstring(imgIdx) + L" in chunk " + std::wstring(chunk.id.begin(), chunk.id.end()));
            }
        }

        // 3. Median stack (sigma-clipping)
        PixelBuffer stacked;
        if (warpedImages.size() > 1)
        {
            if (!MedianStack(warpedImages, stacked))
            {
                Log(L"Median stacking failed for chunk " + std::wstring(chunk.id.begin(), chunk.id.end()));
                return false;
            }
        }
        else if (!warpedImages.empty())
        {
            stacked = std::move(warpedImages[0]);
        }
        else
        {
            return false;
        }

        // 4. Write result to disk via Memory-Mapped File
        std::wstring cachePath = chunk.cache_path;
        if (cachePath.empty())
        {
            // Generate cache path
            cachePath = m_projectManager.GetProjectPath() + L".cache_" +
                std::wstring(chunk.id.begin(), chunk.id.end()) + L".bin";
            chunk.cache_path = cachePath;
        }

        try
        {
            m_cacheManager.WriteChunkToDisk(cachePath, stacked);
        }
        catch (const std::exception& e)
        {
            Log(L"Cache write failed: " + std::wstring(e.what(), e.what() + strlen(e.what())));
            return false;
        }

        return true;
    }

    bool WorkflowController::WarpPerspective(const PixelBuffer& src, PixelBuffer& dst,
                                              const Homography& h, int offsetX, int offsetY,
                                              int chunkW, int chunkH)
    {
        dst.width = chunkW;
        dst.height = chunkH;
        dst.data.resize(chunkW * chunkH * 3, 0);

        // Try GPU path first
        if (m_gpuAvailable && m_cudaPipeline)
        {
            auto result = m_cudaPipeline->WarpPerspective(
                src.data.data(), src.width, src.height,
                dst.data.data(), chunkW, chunkH,
                h.h.data(), offsetX, offsetY);

            if (result == WindowsApp::Compute::ComputeResult::SUCCESS)
                return true;

            Log(L"GPU warp failed, falling back to CPU: " +
                std::wstring(m_cudaPipeline->GetLastError(),
                    m_cudaPipeline->GetLastError() + strlen(m_cudaPipeline->GetLastError())));
        }

        // CPU fallback

        // Compute inverse homography for backward mapping
        // H_inv = adj(H) / det(H)
        const auto& m = h.h;
        float det = m[0] * (m[4] * m[8] - m[5] * m[7])
                  - m[1] * (m[3] * m[8] - m[5] * m[6])
                  + m[2] * (m[3] * m[7] - m[4] * m[6]);

        if (std::abs(det) < 1e-10f) return false;

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

        // Backward mapping: for each output pixel, find source pixel
        for (int y = 0; y < chunkH; y++)
        {
            for (int x = 0; x < chunkW; x++)
            {
                float srcX = invH[0] * (x + offsetX) + invH[1] * (y + offsetY) + invH[2];
                float srcY = invH[3] * (x + offsetX) + invH[4] * (y + offsetY) + invH[5];
                float srcW = invH[6] * (x + offsetX) + invH[7] * (y + offsetY) + invH[8];

                if (std::abs(srcW) < 1e-10f) continue;

                srcX /= srcW;
                srcY /= srcW;

                int sx = static_cast<int>(std::round(srcX));
                int sy = static_cast<int>(std::round(srcY));

                if (sx >= 0 && sx < src.width && sy >= 0 && sy < src.height)
                {
                    int srcIdx = (sy * src.width + sx) * 3;
                    int dstIdx = (y * chunkW + x) * 3;

                    if (srcIdx + 2 < static_cast<int>(src.data.size()) &&
                        dstIdx + 2 < static_cast<int>(dst.data.size()))
                    {
                        dst.data[dstIdx]     = src.data[srcIdx];
                        dst.data[dstIdx + 1] = src.data[srcIdx + 1];
                        dst.data[dstIdx + 2] = src.data[srcIdx + 2];
                    }
                }
            }
        }

        return true;
    }

    bool WorkflowController::MedianStack(std::vector<PixelBuffer>& inputs, PixelBuffer& output)
    {
        if (inputs.empty()) return false;

        output.width = inputs[0].width;
        output.height = inputs[0].height;
        output.data.resize(output.width * output.height * 3);

        int numPixels = output.width * output.height * 3;

        // Try GPU path first
        if (m_gpuAvailable && m_cudaPipeline && inputs.size() <= 32)
        {
            std::vector<const unsigned short*> inputPtrs;
            for (const auto& img : inputs)
                inputPtrs.push_back(img.data.data());

            auto result = m_cudaPipeline->MedianStack(
                inputPtrs.data(), static_cast<int>(inputs.size()),
                output.data.data(), output.width * output.height * 3);

            if (result == WindowsApp::Compute::ComputeResult::SUCCESS)
                return true;

            Log(L"GPU median stack failed, falling back to CPU: " +
                std::wstring(m_cudaPipeline->GetLastError(),
                    m_cudaPipeline->GetLastError() + strlen(m_cudaPipeline->GetLastError())));
        }

        // CPU fallback
        for (int i = 0; i < numPixels; i++)
        {
            // Collect values from all inputs
            std::vector<unsigned short> values;
            values.reserve(inputs.size());

            for (const auto& img : inputs)
            {
                if (i < static_cast<int>(img.data.size()))
                {
                    values.push_back(img.data[i]);
                }
            }

            if (values.empty())
            {
                output.data[i] = 0;
                continue;
            }

            // Sigma-clipping: remove outliers beyond 2 sigma
            if (values.size() > 3)
            {
                float sum = 0, sumSq = 0;
                for (auto v : values) { sum += v; sumSq += v * v; }
                float mean = sum / values.size();
                float variance = sumSq / values.size() - mean * mean;
                float sigma = std::sqrt(std::max(variance, 0.0f));
                float lower = mean - 2.0f * sigma;
                float upper = mean + 2.0f * sigma;

                values.erase(std::remove_if(values.begin(), values.end(),
                    [lower, upper](unsigned short v) {
                        return v < lower || v > upper;
                    }), values.end());
            }

            // Compute median
            std::sort(values.begin(), values.end());
            output.data[i] = values[values.size() / 2];
        }

        return true;
    }

    bool WorkflowController::MultiBandBlend(const PixelBuffer& a, const PixelBuffer& b,
                                             PixelBuffer& output, int blendWidth)
    {
        if (a.width != b.width || a.height != b.height) return false;

        output.width = a.width;
        output.height = a.height;
        output.data.resize(output.width * output.height * 3);

        // Try GPU path first (Laplacian pyramid blending)
        if (m_gpuAvailable && m_cudaPipeline)
        {
            auto result = m_cudaPipeline->MultiBandBlend(
                a.data.data(), b.data.data(), output.data.data(),
                a.width, a.height, 5); // 5 pyramid levels

            if (result == WindowsApp::Compute::ComputeResult::SUCCESS)
                return true;

            Log(L"GPU blend failed, falling back to CPU: " +
                std::wstring(m_cudaPipeline->GetLastError(),
                    m_cudaPipeline->GetLastError() + strlen(m_cudaPipeline->GetLastError())));
        }

        // CPU fallback: simple linear blend
        for (int i = 0; i < static_cast<int>(output.data.size()); i++)
        {
            float t = 0.5f;
            output.data[i] = static_cast<unsigned short>(
                a.data[i] * (1.0f - t) + b.data[i] * t);
        }

        return true;
    }

    // =========================================================================
    // Helpers
    // =========================================================================
    void WorkflowController::Log(const std::wstring& msg)
    {
        if (m_onLog) m_onLog(msg);
    }

    void WorkflowController::UpdateProgress(const std::string& chunkId, float progress)
    {
        m_overallProgress = progress;
        if (m_onProgress) m_onProgress(chunkId, progress);
    }

    void WorkflowController::SetStage(PipelineStage stage)
    {
        m_currentStage = stage;
        if (m_onStage) m_onStage(stage);
    }
}
