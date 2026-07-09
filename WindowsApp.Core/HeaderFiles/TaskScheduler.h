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
        // maxInFlight default (2) matches the original GPU-oriented tuning
        // (see below) - callers on a CPU backend should pass a
        // core-count-derived value instead (e.g.
        // max(1, hardware_concurrency() - 1), since CPU kernels are
        // single-threaded internally and all parallelism comes from this
        // in-flight window - see docs/superpowers/plans for the CPU
        // compute backend's concurrency model).
        explicit TaskScheduler(ProjectManager& projectManager, size_t maxInFlight = 2);

        void RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor);

        // Runs every task for `stage` not already COMPLETED. Returns false
        // if the stage did not fully complete (cancelled, or a task
        // exhausted its retry budget). Never throws for expected failure.
        bool RunStage(PipelineStage stage, CancellationToken token,
                      std::function<void(const Task&, float stageProgress)> onTaskProgress);

    private:
        ProjectManager& m_projectManager;
        std::unordered_map<PipelineStage, std::shared_ptr<ITaskExecutor>> m_executors;
        // Originally a fixed constexpr(2): kept low because Render/Align/
        // Optimize tasks each hold full-resolution GPU buffers for the
        // task's lifetime (see CudaPipeline.cpp) - on an 8GB-class GPU, 4
        // concurrent full-resolution tasks was enough to exhaust VRAM and
        // spill into slower shared system memory. Now an instance field so
        // the composition root can pass a different value for the CPU
        // backend, whose concurrency budget is core-count-, not VRAM-,
        // driven.
        size_t m_maxInFlight;
        static constexpr int kMaxAttempts = 3;
    };
}
