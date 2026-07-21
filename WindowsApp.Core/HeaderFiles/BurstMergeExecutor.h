#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "IComputeBackend.h"
#include <memory>

namespace WindowsApp::Core
{
    // Composite executor registered for BOTH PipelineStage::BURST_MERGE
    // and PipelineStage::BURST_FINISH - dispatches internally by
    // task.stage, the same "one executor, dispatch by a task field"
    // pattern AlignExecutor already uses for unit_kind (here it's by
    // stage instead, since one instance can be registered against
    // multiple PipelineStage keys in TaskScheduler's map). docs/
    // superpowers/plans/2026-07-21-mfnr-block-match-merge.md Task 5.
    //
    // BURST_MERGE: reads every BURST_ALIGN task's decoded frame + (for
    // non-reference frames) TileOffset field, runs
    // IComputeBackend::RobustMergeAccumulate, writes the merged buffer.
    //
    // BURST_FINISH: an identity passthrough for MFNR - copies
    // BURST_MERGE's output blob forward as this task's own output. Real
    // finish-stage processing (sharpen, chroma denoise cleanup) is out of
    // scope for this phase; MFNR has no tone-mapping need (unlike
    // HDR+/Night Sight), so passthrough is a real, minimal, correct
    // implementation here, not a stub.
    //
    // Both stages only run for BurstMode::MFNR - any other mode returns
    // false with a clear "not yet implemented for this BurstMode"
    // task.errorMessage (a genuine failure to surface, not a legitimately-
    // empty case).
    class BurstMergeExecutor : public ITaskExecutor
    {
    public:
        BurstMergeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                            std::shared_ptr<Compute::IComputeBackend> computeBackend);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        bool ExecuteMerge(Task& task);
        bool ExecuteFinish(Task& task);

        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::IComputeBackend> m_computeBackend;
    };
}
