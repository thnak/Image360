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
        static constexpr size_t kMaxInFlight = 4;
        static constexpr int kMaxAttempts = 3;
    };
}
