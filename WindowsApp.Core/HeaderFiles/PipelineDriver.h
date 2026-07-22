#pragma once
#include "Types.h"
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace WindowsApp::Core
{
    class PipelineDriver
    {
    public:
        using ProgressCallback = std::function<void(PipelineStage stage, float overallProgress)>;
        using LogCallback = std::function<void(const std::wstring&)>;
        // Fires once per task as it settles (see TaskScheduler::RunStage's
        // settleFront), after task.status/outputBlobId are already final -
        // a COMPLETED STAGE3_RENDER task's outputBlobId is ready to read
        // and blit immediately, which is what lets a caller show each
        // rendered chunk as it lands instead of only the final assembled
        // result. Runs on whatever thread called Run() (see RunStartButton_Click),
        // never the UI thread directly - callers touching UI state must
        // dispatch back themselves, same as ProgressCallback/LogCallback.
        using TaskCallback = std::function<void(PipelineStage stage, const Task& task)>;

        // maxInFlight default (2) matches TaskScheduler's own GPU-oriented
        // default - callers on a CPU backend should pass a core-count-
        // derived value instead (CPU kernels are single-threaded
        // internally, so all parallelism comes from this in-flight
        // window). onTaskCompleted defaults to nullptr since most callers
        // (tests, etc.) only care about stage-level progress.
        void Initialize(ProgressCallback onProgress, LogCallback onLog, size_t maxInFlight = 2,
                         TaskCallback onTaskCompleted = nullptr);
        void RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor);

        // Drives STAGE0_INGEST -> STAGE1_ALIGN -> STAGE2_OPTIMIZE ->
        // STAGE3_RENDER in order, skipping stages whose tasks are already
        // fully COMPLETED. Safe to call again after a previous cancelled
        // run or a crash - every call is a resume.
        bool Run(ProjectManager& projectManager, CancellationToken token);

        PipelineStage GetCurrentStage() const;
        float GetOverallProgress() const;

    private:
        // Executors are registered before a ProjectManager is known (e.g.
        // at app startup), so they're held here and handed to a
        // TaskScheduler constructed locally inside Run() against whichever
        // ProjectManager the caller passes in.
        std::unordered_map<PipelineStage, std::shared_ptr<ITaskExecutor>> m_executors;
        ProgressCallback m_onProgress;
        LogCallback m_onLog;
        TaskCallback m_onTaskCompleted;
        size_t m_maxInFlight = 2;
        std::atomic<PipelineStage> m_currentStage{ PipelineStage::IDLE };
        std::atomic<float> m_overallProgress{ 0.0f };
    };
}
