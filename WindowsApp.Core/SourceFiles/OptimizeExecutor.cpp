#include "pch.h"
#include "HeaderFiles/OptimizeExecutor.h"

namespace WindowsApp::Core
{
    OptimizeExecutor::OptimizeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                        std::shared_ptr<Compute::CudaPipeline> cudaPipeline,
                                        std::shared_ptr<Compute::NvJpegCodec> nvJpegCodec)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_cudaPipeline(std::move(cudaPipeline))
        , m_nvJpegCodec(std::move(nvJpegCodec))
    {
    }

    bool OptimizeExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;

        if (task.unitKind == "gain") return ExecuteGain(task, token);
        if (task.unitKind == "color") return ExecuteColorTransfer(task, token);
        if (task.unitKind == "ba_checkpoint") return ExecuteBundleAdjustment(task, token);

        return false;
    }

    // Implemented in a follow-up issue
    // (docs/superpowers/plans/2026-07-07-optimize-stage.md Task 2).
    bool OptimizeExecutor::ExecuteGain(Task& /* task */, CancellationToken /* token */)
    {
        return false;
    }

    // Implemented in a follow-up issue (Task 3).
    bool OptimizeExecutor::ExecuteColorTransfer(Task& /* task */, CancellationToken /* token */)
    {
        return false;
    }

    // Implemented in a follow-up issue (Task 4).
    bool OptimizeExecutor::ExecuteBundleAdjustment(Task& /* task */, CancellationToken /* token */)
    {
        return false;
    }
}
