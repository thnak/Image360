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
    // 2026-07-21-hdrplus-tile-fft-merge.md Task 3, for Super Res Zoom by
    // docs/superpowers/plans/2026-07-21-superres-structure-tensor-merge.md
    // Task 3, and for Night Sight by docs/superpowers/plans/
    // 2026-07-22-night-sight.md Tasks 3-5.
    //
    // BURST_MERGE: reads every BURST_ALIGN task's decoded frame + (for
    // non-reference frames) its TileOffset/TileOffsetF field, then runs
    // IComputeBackend::RobustMergeAccumulate (MFNR), ::TileFftMerge
    // (HDR_PLUS), or ::StructureTensorKernelRegression (SUPER_RES/
    // NIGHT_SIGHT) - three genuinely different merge algorithms (not four -
    // Night Sight reuses Super Res Zoom's merge kernel, per
    // docs/COMPUTATIONAL_PHOTOGRAPHY.md SS2.2), not parameter variants,
    // per SS2.3 - and writes the merged buffer (upsampled by
    // kSuperResScaleFactor for SUPER_RES, native resolution for
    // NIGHT_SIGHT). NIGHT_SIGHT additionally runs Kernels::MeterMotion
    // first to drop unusably-shifted frames and adapt the merge's
    // noiseVariance (docs/superpowers/plans/2026-07-22-night-sight.md
    // Architecture SS1/3).
    //
    // BURST_FINISH: MFNR and SUPER_RES are both identity passthroughs -
    // copy BURST_MERGE's output blob forward as this task's own output
    // (neither has a tone-mapping need in this phase's scope, so
    // passthrough is a real, minimal, correct implementation, not a stub).
    // HDR_PLUS is a real transform - two synthetic tone-curve exposures of
    // the merged image, fused via Kernels::ExposureFusion::FuseTwoExposures
    // (exposure-fusion tone mapping, docs/COMPUTATIONAL_PHOTOGRAPHY.md
    // SS2.1's Finish stage). NIGHT_SIGHT is also a real transform - a
    // single-image "painterly" S-curve + vignette via
    // Kernels::PainterlyToneCurve::Apply, distinct from HDR_PLUS's
    // multi-exposure fusion (SS2.2's Finish stage).
    //
    // Both stages only run for BurstMode::MFNR, HDR_PLUS, SUPER_RES, or
    // NIGHT_SIGHT - any other mode returns false with a clear "not yet
    // implemented for this BurstMode" task.errorMessage (a genuine failure
    // to surface, not a legitimately-empty case).
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
