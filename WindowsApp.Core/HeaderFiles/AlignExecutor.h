#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "CudaPipeline.h"
#include "NvJpegCodec.h"
#include <memory>

namespace WindowsApp::Core
{
    // Composite executor registered once for STAGE1_ALIGN - TaskScheduler
    // maps one executor per PipelineStage, so this dispatches internally
    // by task.unitKind ("image" -> feature extraction, "pair" -> match +
    // RANSAC) instead of registering two separate executors, which would
    // silently overwrite each other in TaskScheduler's registration map.
    class AlignExecutor : public ITaskExecutor
    {
    public:
        AlignExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                      std::shared_ptr<Compute::CudaPipeline> cudaPipeline,
                      std::shared_ptr<Compute::NvJpegCodec> nvJpegCodec);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        bool ExecuteFeatureExtraction(Task& task, CancellationToken token);
        bool ExecuteMatch(Task& task, CancellationToken token);

        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::CudaPipeline> m_cudaPipeline;
        std::shared_ptr<Compute::NvJpegCodec> m_nvJpegCodec;
    };
}
