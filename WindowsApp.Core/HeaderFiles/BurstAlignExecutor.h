#pragma once
#include "ITaskExecutor.h"
#include "ProjectManager.h"
#include "StorageEngine.h"
#include "IComputeBackend.h"
#include "IImageCodec.h"
#include <memory>

namespace WindowsApp::Core
{
    // Registered for PipelineStage::BURST_ALIGN - docs/superpowers/plans/
    // 2026-07-21-mfnr-block-match-merge.md Task 4. unit_kind = "frame",
    // unit_key = the input image's DB id (burst-mode frames are stored via
    // the same input_images table/AddInputImage panorama input images use).
    //
    // Decodes this task's own frame (reusing DecodeInputImage, shared with
    // RawIngestExecutor) and writes it as this task's output blob. The
    // frame with the lowest id (the first one added) is the alignment
    // reference and needs no further work; every other frame ALSO decodes
    // the reference frame (a redundant per-task decode - acceptable at
    // MFNR's typical burst sizes, see the plan's SS9) and runs
    // BlockMatchAlign against it, serializing the resulting TileOffset
    // field into task.checkpointJson (Kernels::SerializeTileOffsets).
    class BurstAlignExecutor : public ITaskExecutor
    {
    public:
        BurstAlignExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                            std::shared_ptr<Compute::IComputeBackend> computeBackend,
                            std::shared_ptr<Compute::IImageCodec> jpegCodec);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::IComputeBackend> m_computeBackend;
        std::shared_ptr<Compute::IImageCodec> m_jpegCodec;
    };
}
