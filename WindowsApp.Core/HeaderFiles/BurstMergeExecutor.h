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
    // superpowers/plans/2026-07-21-mfnr-block-match-merge.md Task 5,
    // extended for HDR+ by docs/superpowers/plans/
    // 2026-07-21-hdrplus-tile-fft-merge.md Task 3.
    //
    // BURST_MERGE: reads every BURST_ALIGN task's decoded frame + (for
    // non-reference frames) TileOffset field, then runs
    // IComputeBackend::RobustMergeAccumulate (MFNR) or ::TileFftMerge
    // (HDR_PLUS - a genuinely different merge algorithm, not a parameter
    // variant, per docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.3), writes the
    // merged buffer.
    //
    // BURST_FINISH: MFNR is an identity passthrough - copies BURST_MERGE's
    // output blob forward as this task's own output (MFNR has no
    // tone-mapping need, so passthrough is a real, minimal, correct
    // implementation, not a stub). HDR_PLUS is a real transform - two
    // synthetic tone-curve exposures of the merged image, fused via
    // Kernels::ExposureFusion::FuseTwoExposures (exposure-fusion tone
    // mapping, docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.1's Finish stage).
    //
    // Both stages only run for BurstMode::MFNR or BurstMode::HDR_PLUS -
    // any other mode returns false with a clear "not yet implemented for
    // this BurstMode" task.errorMessage (a genuine failure to surface, not
    // a legitimately-empty case).
    class BurstMergeExecutor : public ITaskExecutor
    {
    public:
        BurstMergeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                            std::shared_ptr<Compute::IComputeBackend> computeBackend);

        bool Execute(Task& task, CancellationToken token) override;

    private:
        // Frames + per-tile offsets collected from completed BURST_ALIGN
        // tasks - shared gathering logic for both merge algorithms (MFNR's
        // RobustMergeAccumulate and HDR_PLUS's TileFftMerge take the exact
        // same frame/offset shape).
        struct GatheredFrames
        {
            PixelBuffer referenceBuffer;
            std::vector<PixelBuffer> nonReferenceBuffers;
            std::vector<std::vector<Compute::TileOffset>> perFrameOffsets;
            int tilesX = 0;
            int tilesY = 0;
        };
        bool GatherAlignedFrames(Task& task, GatheredFrames& out);

        bool ExecuteMerge(Task& task);
        bool ExecuteFinish(Task& task);

        ProjectManager& m_projectManager;
        StorageEngine& m_storageEngine;
        std::shared_ptr<Compute::IComputeBackend> m_computeBackend;
    };
}
