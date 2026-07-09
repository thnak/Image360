#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "IComputeBackend.h"
#include "IImageCodec.h"
#include <memory>

namespace WindowsApp::Core
{
    // Composite executor registered once for STAGE2_OPTIMIZE - same
    // dispatch-by-unitKind pattern as AlignExecutor, since TaskScheduler
    // maps one executor per PipelineStage and this stage has three kinds
    // of work ("gain", "color", "ba_checkpoint").
    class OptimizeExecutor : public ITaskExecutor
    {
    public:
        OptimizeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                          std::shared_ptr<Compute::IComputeBackend> cudaPipeline,
                          std::shared_ptr<Compute::IImageCodec> nvJpegCodec);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        bool ExecuteGain(Task& task, CancellationToken token);            // "gain"
        bool ExecuteColorTransfer(Task& task, CancellationToken token);   // "color"
        bool ExecuteBundleAdjustment(Task& task, CancellationToken token); // "ba_checkpoint"

        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::IComputeBackend> m_cudaPipeline;
        std::shared_ptr<Compute::IImageCodec> m_nvJpegCodec;
    };
}
