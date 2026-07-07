#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "CudaPipeline.h"
#include <memory>

namespace WindowsApp::Core
{
    // First real (non-stub) ITaskExecutor - docs/ARCHITECTURE.md SS4.1.
    // unit_kind = "image", unit_key = the input image's DB id.
    class RawIngestExecutor : public ITaskExecutor
    {
    public:
        RawIngestExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                           std::shared_ptr<Compute::CudaPipeline> cudaPipeline);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::CudaPipeline> m_cudaPipeline;
    };
}
