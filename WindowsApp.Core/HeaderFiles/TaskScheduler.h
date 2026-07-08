#pragma once
#include "Types.h"
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include <unordered_map>
#include <memory>
#include <functional>

namespace WindowsApp::Core
{
    class TaskScheduler
    {
    public:
        explicit TaskScheduler(ProjectManager& projectManager);

        void RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor);

        // Runs every task for `stage` not already COMPLETED. Returns false
        // if the stage did not fully complete (cancelled, or a task
        // exhausted its retry budget). Never throws for expected failure.
        bool RunStage(PipelineStage stage, CancellationToken token,
                      std::function<void(const Task&, float stageProgress)> onTaskProgress);

    private:
        ProjectManager& m_projectManager;
        std::unordered_map<PipelineStage, std::shared_ptr<ITaskExecutor>> m_executors;
        // Kept low because Render/Align/Optimize tasks each hold full-resolution
        // GPU buffers for the task's lifetime (see CudaPipeline.cpp) - on an
        // 8GB-class GPU, 4 concurrent full-resolution tasks was enough to
        // exhaust VRAM and spill into slower shared system memory.
        static constexpr size_t kMaxInFlight = 2;
        static constexpr int kMaxAttempts = 3;
    };
}
