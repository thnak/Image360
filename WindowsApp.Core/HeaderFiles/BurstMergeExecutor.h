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
    // 2026-07-21-hdrplus-tile-fft-merge.md Task 3, and for Super Res Zoom
    // by docs/superpowers/plans/2026-07-21-superres-structure-tensor-merge.md
    // Task 3.
    //
    // BURST_MERGE: reads every BURST_ALIGN task's decoded frame + (for
    // non-reference frames) its TileOffset/TileOffsetF field, then runs
    // IComputeBackend::RobustMergeAccumulate (MFNR), ::TileFftMerge
    // (HDR_PLUS), or ::StructureTensorKernelRegression (SUPER_RES) - three
    // genuinely different merge algorithms, not parameter variants, per
    // docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.3 - and writes the merged
    // buffer (upsampled, for SUPER_RES).
    //
    // BURST_FINISH: MFNR and SUPER_RES are both identity passthroughs -
    // copy BURST_MERGE's output blob forward as this task's own output
    // (neither has a tone-mapping need in this phase's scope, so
    // passthrough is a real, minimal, correct implementation, not a stub).
    // HDR_PLUS is a real transform - two synthetic tone-curve exposures of
    // the merged image, fused via Kernels::ExposureFusion::FuseTwoExposures
    // (exposure-fusion tone mapping, docs/COMPUTATIONAL_PHOTOGRAPHY.md
    // SS2.1's Finish stage).
    //
    // Both stages only run for BurstMode::MFNR, HDR_PLUS, or SUPER_RES -
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
        // tasks - shared gathering logic for all three merge algorithms.
        // Only one of perFrameOffsets/perFrameOffsetsF is populated,
        // depending on GetBurstMode() (MFNR/HDR_PLUS use the integer
        // BlockMatchAlign offsets directly; SUPER_RES uses the sub-pixel-
        // refined ones BurstAlignExecutor serializes for that mode).
        struct GatheredFrames
        {
            PixelBuffer referenceBuffer;
            std::vector<PixelBuffer> nonReferenceBuffers;
            std::vector<std::vector<Compute::TileOffset>> perFrameOffsets;
            std::vector<std::vector<Compute::TileOffsetF>> perFrameOffsetsF;
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
