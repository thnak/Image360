#include "pch.h"
#include "HeaderFiles/BurstMergeExecutor.h"
#include "HeaderFiles/BurstCommon.h"
#include "HeaderFiles/BlockMatchAlignKernel.h"
#include "HeaderFiles/SubPixelRefineKernel.h"
#include "HeaderFiles/ExposureFusionKernel.h"
#include "HeaderFiles/NightSightMotionMeter.h"
#include "HeaderFiles/PainterlyToneCurveKernel.h"

#include <cmath>
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    BurstMergeExecutor::BurstMergeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                            std::shared_ptr<Compute::IComputeBackend> computeBackend)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_computeBackend(std::move(computeBackend))
    {
    }

    bool BurstMergeExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;

        BurstMode mode = m_projectManager.GetBurstMode();
        if (mode != BurstMode::MFNR && mode != BurstMode::HDR_PLUS && mode != BurstMode::SUPER_RES
            && mode != BurstMode::NIGHT_SIGHT)
        {
            task.errorMessage = "BurstMergeExecutor: not yet implemented for this BurstMode (only MFNR/HDR_PLUS/SUPER_RES/NIGHT_SIGHT are implemented).";
            return false;
        }

        switch (task.stage)
        {
        case PipelineStage::BURST_MERGE:  return ExecuteMerge(task);
        case PipelineStage::BURST_FINISH: return ExecuteFinish(task);
        default: return false;
        }
    }

    bool BurstMergeExecutor::GatherAlignedFrames(Task& task, GatheredFrames& out)
    {
        std::vector<Task> alignTasks = m_projectManager.GetTasksForStage(PipelineStage::BURST_ALIGN);
        if (alignTasks.empty())
        {
            task.errorMessage = "No BURST_ALIGN tasks found.";
            return false;
        }

        int referenceId = BurstReferenceImageId(m_projectManager.GetInputImages());

        const Task* referenceTask = nullptr;
        std::vector<const Task*> nonReferenceTasks;
        for (const auto& alignTask : alignTasks)
        {
            if (alignTask.status != TaskStatus::COMPLETED || !alignTask.outputBlobId.has_value())
            {
                task.errorMessage = "A BURST_ALIGN task has not completed yet.";
                return false;
            }

            int imageId = 0;
            try { imageId = std::stoi(alignTask.unitKey); }
            catch (...) { task.errorMessage = "Malformed BURST_ALIGN unitKey."; return false; }

            if (imageId == referenceId) referenceTask = &alignTask;
            else nonReferenceTasks.push_back(&alignTask);
        }
        if (!referenceTask)
        {
            task.errorMessage = "Reference frame's BURST_ALIGN task not found.";
            return false;
        }

        auto referenceBuffer = m_storageEngine.ReadPixelBuffer(*referenceTask->outputBlobId);
        if (!referenceBuffer.has_value())
        {
            task.errorMessage = "Failed to read the reference frame's decoded blob.";
            return false;
        }
        out.referenceBuffer = std::move(*referenceBuffer);

        out.tilesX = (out.referenceBuffer.width + kBurstTileSize - 1) / kBurstTileSize;
        out.tilesY = (out.referenceBuffer.height + kBurstTileSize - 1) / kBurstTileSize;

        out.nonReferenceBuffers.reserve(nonReferenceTasks.size());
        BurstMode gatherMode = m_projectManager.GetBurstMode();
        bool useSubPixelOffsets = gatherMode == BurstMode::SUPER_RES || gatherMode == BurstMode::NIGHT_SIGHT;
        if (useSubPixelOffsets) out.perFrameOffsetsF.reserve(nonReferenceTasks.size());
        else out.perFrameOffsets.reserve(nonReferenceTasks.size());

        for (const Task* nonRefTask : nonReferenceTasks)
        {
            auto buffer = m_storageEngine.ReadPixelBuffer(*nonRefTask->outputBlobId);
            if (!buffer.has_value())
            {
                task.errorMessage = "Failed to read a burst frame's decoded blob.";
                return false;
            }
            if (buffer->width != out.referenceBuffer.width || buffer->height != out.referenceBuffer.height)
            {
                task.errorMessage = "Burst frame dimensions don't match the reference frame.";
                return false;
            }

            if (useSubPixelOffsets)
            {
                std::vector<Compute::TileOffsetF> offsetsF;
                int tilesX = 0, tilesY = 0;
                if (!Kernels::DeserializeTileOffsetsF(nonRefTask->checkpointJson, offsetsF, tilesX, tilesY)
                    || tilesX != out.tilesX || tilesY != out.tilesY)
                {
                    task.errorMessage = "Failed to deserialize (or inconsistent) TileOffsetF field for a burst frame.";
                    return false;
                }
                out.perFrameOffsetsF.push_back(std::move(offsetsF));
            }
            else
            {
                std::vector<Compute::TileOffset> offsets;
                int tilesX = 0, tilesY = 0;
                if (!Kernels::DeserializeTileOffsets(nonRefTask->checkpointJson, offsets, tilesX, tilesY)
                    || tilesX != out.tilesX || tilesY != out.tilesY)
                {
                    task.errorMessage = "Failed to deserialize (or inconsistent) TileOffset field for a burst frame.";
                    return false;
                }
                out.perFrameOffsets.push_back(std::move(offsets));
            }

            out.nonReferenceBuffers.push_back(std::move(*buffer));
        }

        return true;
    }

    bool BurstMergeExecutor::ExecuteMerge(Task& task)
    {
        GatheredFrames gathered;
        if (!GatherAlignedFrames(task, gathered)) return false;
        if (!m_computeBackend) return false;

        std::vector<const unsigned short*> framePtrs;
        framePtrs.reserve(1 + gathered.nonReferenceBuffers.size());
        framePtrs.push_back(gathered.referenceBuffer.data.data());
        for (const auto& buffer : gathered.nonReferenceBuffers) framePtrs.push_back(buffer.data.data());

        BurstMode mode = m_projectManager.GetBurstMode();
        Compute::ComputeResult result;
        PixelBuffer merged;

        if (mode == BurstMode::SUPER_RES || mode == BurstMode::NIGHT_SIGHT)
        {
            // NIGHT_SIGHT-only: meter motion over the non-reference frames
            // and drop unusably-shifted ones before building framePtrs/
            // offsetPtrsF (docs/superpowers/plans/2026-07-22-night-sight.md
            // Architecture SS1/3) - SUPER_RES keeps every frame, matching
            // its existing unchanged behavior.
            std::vector<bool> keepFrame(gathered.perFrameOffsetsF.size(), true);
            float noiseVariance = kSuperResNoiseVariance;
            int scaleFactor = kSuperResScaleFactor;

            if (mode == BurstMode::NIGHT_SIGHT)
            {
                Kernels::MotionMeteringResult metering =
                    Kernels::MeterMotion(gathered.perFrameOffsetsF, kNightSightBaseNoiseVariance);
                keepFrame = metering.keepFrame;
                noiseVariance = metering.noiseVariance;
                scaleFactor = kNightSightScaleFactor;
            }

            std::vector<const unsigned short*> filteredFramePtrs;
            std::vector<const Compute::TileOffsetF*> offsetPtrsF;
            filteredFramePtrs.reserve(framePtrs.size());
            offsetPtrsF.reserve(gathered.perFrameOffsetsF.size());
            filteredFramePtrs.push_back(gathered.referenceBuffer.data.data());
            for (size_t i = 0; i < gathered.perFrameOffsetsF.size(); ++i)
            {
                if (!keepFrame[i]) continue;
                filteredFramePtrs.push_back(gathered.nonReferenceBuffers[i].data.data());
                offsetPtrsF.push_back(gathered.perFrameOffsetsF[i].data());
            }

            merged.width = gathered.referenceBuffer.width * scaleFactor;
            merged.height = gathered.referenceBuffer.height * scaleFactor;
            merged.data.resize(static_cast<size_t>(merged.width) * merged.height * 3);

            result = m_computeBackend->StructureTensorKernelRegression(
                filteredFramePtrs.data(), static_cast<int>(filteredFramePtrs.size()), offsetPtrsF.data(),
                gathered.referenceBuffer.width, gathered.referenceBuffer.height, kBurstTileSize,
                gathered.tilesX, gathered.tilesY, scaleFactor, noiseVariance,
                merged.data.data());
        }
        else
        {
            std::vector<const Compute::TileOffset*> offsetPtrs;
            offsetPtrs.reserve(gathered.perFrameOffsets.size());
            for (const auto& offsets : gathered.perFrameOffsets) offsetPtrs.push_back(offsets.data());

            merged.width = gathered.referenceBuffer.width;
            merged.height = gathered.referenceBuffer.height;
            merged.data.resize(gathered.referenceBuffer.data.size());

            if (mode == BurstMode::HDR_PLUS)
            {
                result = m_computeBackend->TileFftMerge(
                    framePtrs.data(), static_cast<int>(framePtrs.size()), offsetPtrs.data(),
                    merged.width, merged.height, kBurstTileSize, gathered.tilesX, gathered.tilesY,
                    kHdrPlusNoiseVariance, merged.data.data());
            }
            else
            {
                result = m_computeBackend->RobustMergeAccumulate(
                    framePtrs.data(), static_cast<int>(framePtrs.size()), offsetPtrs.data(),
                    merged.width, merged.height, kBurstTileSize, gathered.tilesX, gathered.tilesY,
                    kBurstMergeSigma, merged.data.data());
            }
        }
        if (result != Compute::ComputeResult::SUCCESS)
        {
            task.errorMessage = m_computeBackend->GetLastError();
            return false;
        }

        auto blobId = m_storageEngine.WritePixelBuffer(merged, "raw_rgb48");
        if (!blobId.has_value())
        {
            task.errorMessage = "Failed to write the merged buffer.";
            return false;
        }

        task.outputBlobId = blobId;
        return true;
    }

    namespace
    {
        // v' = 65535*(v/65535)^gamma, per channel.
        void ApplyToneCurve(const PixelBuffer& src, float gamma, PixelBuffer& outDst)
        {
            outDst.width = src.width;
            outDst.height = src.height;
            outDst.data.resize(src.data.size());
            for (size_t i = 0; i < src.data.size(); ++i)
            {
                float normalized = static_cast<float>(src.data[i]) / 65535.0f;
                float mapped = std::pow((std::max)(0.0f, normalized), gamma) * 65535.0f;
                outDst.data[i] = static_cast<unsigned short>((std::max)(0.0f, (std::min)(65535.0f, mapped)) + 0.5f);
            }
        }
    }

    bool BurstMergeExecutor::ExecuteFinish(Task& task)
    {
        std::vector<Task> mergeTasks = m_projectManager.GetTasksForStage(PipelineStage::BURST_MERGE);
        if (mergeTasks.size() != 1 || mergeTasks[0].status != TaskStatus::COMPLETED
            || !mergeTasks[0].outputBlobId.has_value())
        {
            task.errorMessage = "BURST_MERGE task has not completed yet.";
            return false;
        }

        if (m_projectManager.GetBurstMode() == BurstMode::HDR_PLUS)
        {
            auto mergedBuffer = m_storageEngine.ReadPixelBuffer(*mergeTasks[0].outputBlobId);
            if (!mergedBuffer.has_value())
            {
                task.errorMessage = "Failed to read BURST_MERGE's output PixelBuffer.";
                return false;
            }

            PixelBuffer brightExposure, darkExposure;
            ApplyToneCurve(*mergedBuffer, kHdrPlusBrightGamma, brightExposure);
            ApplyToneCurve(*mergedBuffer, kHdrPlusDarkGamma, darkExposure);

            PixelBuffer fused;
            fused.width = mergedBuffer->width;
            fused.height = mergedBuffer->height;
            Kernels::ExposureFusion::FuseTwoExposures(
                brightExposure.data.data(), darkExposure.data.data(),
                fused.width, fused.height, Kernels::ExposureFusion::kDefaultNumBands, fused.data);

            auto blobId = m_storageEngine.WritePixelBuffer(fused, "raw_rgb48");
            if (!blobId.has_value())
            {
                task.errorMessage = "Failed to write BURST_FINISH's tone-mapped output.";
                return false;
            }

            task.outputBlobId = blobId;
            return true;
        }

        if (m_projectManager.GetBurstMode() == BurstMode::NIGHT_SIGHT)
        {
            auto mergedBuffer = m_storageEngine.ReadPixelBuffer(*mergeTasks[0].outputBlobId);
            if (!mergedBuffer.has_value())
            {
                task.errorMessage = "Failed to read BURST_MERGE's output PixelBuffer.";
                return false;
            }

            PixelBuffer toneMapped;
            toneMapped.width = mergedBuffer->width;
            toneMapped.height = mergedBuffer->height;
            Kernels::PainterlyToneCurve::Apply(
                mergedBuffer->data.data(), mergedBuffer->width, mergedBuffer->height,
                kNightSightShadowGamma, kNightSightHighlightRolloff, kNightSightVignetteStrength,
                toneMapped.data);

            auto blobId = m_storageEngine.WritePixelBuffer(toneMapped, "raw_rgb48");
            if (!blobId.has_value())
            {
                task.errorMessage = "Failed to write BURST_FINISH's tone-mapped output.";
                return false;
            }

            task.outputBlobId = blobId;
            return true;
        }

        auto bytes = m_storageEngine.ReadBlob(*mergeTasks[0].outputBlobId);
        if (!bytes.has_value())
        {
            task.errorMessage = "Failed to read BURST_MERGE's output blob.";
            return false;
        }

        // Passthrough: copies the exact bytes (including WritePixelBuffer's
        // embedded width/height header) forward under a new blob, rather
        // than round-tripping through PixelBuffer - only MFNR's and
        // SUPER_RES's BURST_FINISH have no actual transformation to apply
        // in this phase's scope (see this class's header comment).
        auto blobId = m_storageEngine.WriteBlob(bytes->data(), bytes->size(), "raw_rgb48");
        if (!blobId.has_value())
        {
            task.errorMessage = "Failed to write BURST_FINISH's output blob.";
            return false;
        }

        task.outputBlobId = blobId;
        return true;
    }
}
