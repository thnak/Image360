#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "CudaPipeline.h"
#include <memory>

namespace WindowsApp::Core
{
    // unit_kind = "chunk", unit_key = chunk id (e.g. "C_4_2"). Single
    // unit kind, so unlike Align/Optimize this needs no composite
    // dispatch-by-unitKind.
    //
    // Note: chunks.status/cache_path (the v1 columns, still present in
    // the schema) are NOT the resume mechanism anymore - the generic
    // tasks table is (docs/ARCHITECTURE.md SS4.4). This executor does
    // not read or write chunks.status/cache_path at all.
    class RenderExecutor : public ITaskExecutor
    {
    public:
        RenderExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                        std::shared_ptr<Compute::CudaPipeline> cudaPipeline);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::CudaPipeline> m_cudaPipeline;
    };
}
