#pragma once
#include "Types.h"
#include "ProjectManager.h"
#include "CacheManager.h"
#include <string>
#include <functional>
#include <stop_token>
#include <mutex>
#include <atomic>
#include <memory>

// Forward declare to avoid CUDA header dependency in Core
namespace WindowsApp::Compute { class CudaPipeline; }

namespace WindowsApp::Core
{
    enum class PipelineStage
    {
        IDLE,
        STAGE1_ALIGN,
        STAGE2_OPTIMIZE,
        STAGE3_RENDER,
        COMPLETED,
        CANCELLED,
        FAILED
    };

    class WorkflowController
    {
    public:
        using ProgressCallback = std::function<void(const std::string& chunkId, float progress)>;
        using LogCallback = std::function<void(const std::wstring& message)>;
        using StageCallback = std::function<void(PipelineStage stage)>;

        void Initialize(ProgressCallback onProgress, LogCallback onLog, StageCallback onStage);
        bool StartProcessing(std::stop_token stopToken, const std::wstring& projectPath);
        void Cancel();

        PipelineStage GetCurrentStage() const { return m_currentStage; }
        float GetOverallProgress() const { return m_overallProgress; }

    private:
        ProgressCallback m_onProgress;
        LogCallback m_onLog;
        StageCallback m_onStage;
        std::mutex m_mutex;

        PipelineStage m_currentStage = PipelineStage::IDLE;
        std::atomic<float> m_overallProgress{ 0.0f };

        ProjectManager m_projectManager;
        CacheManager m_cacheManager;
        std::unique_ptr<WindowsApp::Compute::CudaPipeline> m_cudaPipeline;
        bool m_gpuAvailable = false;

        // Stage 1: Low-res alignment
        bool Stage1_Align(std::stop_token& stopToken);
        bool LoadImageLowRes(const std::wstring& path, PixelBuffer& output, int scalePercent);
        bool ExtractFeatures(const PixelBuffer& image, std::vector<std::array<float, 2>>& keypoints);
        bool MatchFeatures(const std::vector<std::array<float, 2>>& kp1,
                           const std::vector<std::array<float, 2>>& kp2,
                           std::vector<std::pair<int, int>>& matches);
        bool ComputeHomographyRANSAC(const std::vector<std::array<float, 2>>& pts1,
                                      const std::vector<std::array<float, 2>>& pts2,
                                      Homography& outH);

        // Stage 2: Global optimization
        bool Stage2_Optimize(std::stop_token& stopToken);
        bool ComputeGainCompensation();
        bool ApplyColorTransfer();
        bool RunBundleAdjustment();

        // Stage 3: High-res tile rendering
        bool Stage3_Render(std::stop_token& stopToken);
        bool ProcessChunk(ChunkModel& chunk, std::stop_token& stopToken);
        bool WarpPerspective(const PixelBuffer& src, PixelBuffer& dst, const Homography& h,
                             int offsetX, int offsetY, int chunkW, int chunkH);
        bool MedianStack(std::vector<PixelBuffer>& inputs, PixelBuffer& output);
        bool MultiBandBlend(const PixelBuffer& a, const PixelBuffer& b, PixelBuffer& output, int blendWidth);

        // Helpers
        void Log(const std::wstring& msg);
        void UpdateProgress(const std::string& chunkId, float progress);
        void SetStage(PipelineStage stage);
    };
}
