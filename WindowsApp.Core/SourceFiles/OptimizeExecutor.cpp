#include "pch.h"
#include "HeaderFiles/OptimizeExecutor.h"
#include "HeaderFiles/ImageLoader.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        // The whole-project reference image is the smallest input-image
        // id (SQLite AUTOINCREMENT starts at 1, not 0 - matches the
        // "smallest id is A" ordering AlignExecutor's SeedAlignTasks
        // already relies on, since GetInputImages() is id-ascending).
        std::optional<int> ReferenceImageId(ProjectManager& projectManager)
        {
            const auto& images = projectManager.GetInputImages();
            if (images.empty()) return std::nullopt;
            return images.front().id;
        }

        // Same feature-blob framing as AlignExecutor.cpp - duplicated
        // locally rather than shared, since it's a small, self-contained
        // helper and no other file in this codebase currently shares
        // such helpers across executor translation units.
        bool DeserializeFeatures(
            const std::vector<uint8_t>& bytes,
            std::vector<Compute::FeaturePoint>& outPoints,
            std::vector<uint64_t>& outDescriptorWords)
        {
            if (bytes.size() < sizeof(int32_t)) return false;

            int32_t count = 0;
            std::memcpy(&count, bytes.data(), sizeof(count));
            if (count < 0) return false;

            size_t expected = sizeof(count)
                + static_cast<size_t>(count) * sizeof(Compute::FeaturePoint)
                + static_cast<size_t>(count) * 4 * sizeof(uint64_t);
            if (bytes.size() < expected) return false;

            outPoints.resize(count);
            outDescriptorWords.resize(static_cast<size_t>(count) * 4);

            size_t offset = sizeof(count);
            if (count > 0)
            {
                std::memcpy(outPoints.data(), bytes.data() + offset, count * sizeof(Compute::FeaturePoint));
                offset += count * sizeof(Compute::FeaturePoint);
                std::memcpy(outDescriptorWords.data(), bytes.data() + offset, static_cast<size_t>(count) * 4 * sizeof(uint64_t));
            }
            return true;
        }

        std::optional<int64_t> FindAlignFeatureBlobId(ProjectManager& projectManager, int imageId)
        {
            std::string key = std::to_string(imageId);
            for (const auto& task : projectManager.GetTasksForStage(PipelineStage::STAGE1_ALIGN))
            {
                if (task.unitKind == "image" && task.unitKey == key && task.status == TaskStatus::COMPLETED)
                {
                    return task.outputBlobId;
                }
            }
            return std::nullopt;
        }

        // Finds the STAGE1_ALIGN/"pair" task between two images (either
        // key order) and returns it if COMPLETED.
        std::optional<Task> FindCompletedAlignPair(ProjectManager& projectManager, int imageA, int imageB)
        {
            std::string keyForward = std::to_string(imageA) + ":" + std::to_string(imageB);
            std::string keyBackward = std::to_string(imageB) + ":" + std::to_string(imageA);
            for (const auto& task : projectManager.GetTasksForStage(PipelineStage::STAGE1_ALIGN))
            {
                if (task.unitKind == "pair" && task.status == TaskStatus::COMPLETED &&
                    (task.unitKey == keyForward || task.unitKey == keyBackward))
                {
                    return task;
                }
            }
            return std::nullopt;
        }

        constexpr int kMaxMatchesForGain = 2000;

        std::optional<int64_t> FindIngestBlobId(ProjectManager& projectManager, int imageId)
        {
            std::string key = std::to_string(imageId);
            for (const auto& task : projectManager.GetTasksForStage(PipelineStage::STAGE0_INGEST))
            {
                if (task.unitKind == "image" && task.unitKey == key && task.status == TaskStatus::COMPLETED)
                {
                    return task.outputBlobId;
                }
            }
            return std::nullopt;
        }
    }

    OptimizeExecutor::OptimizeExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                        std::shared_ptr<Compute::CudaPipeline> cudaPipeline,
                                        std::shared_ptr<Compute::NvJpegCodec> nvJpegCodec)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_cudaPipeline(std::move(cudaPipeline))
        , m_nvJpegCodec(std::move(nvJpegCodec))
    {
    }

    bool OptimizeExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;

        if (task.unitKind == "gain") return ExecuteGain(task, token);
        if (task.unitKind == "color") return ExecuteColorTransfer(task, token);
        if (task.unitKind == "ba_checkpoint") return ExecuteBundleAdjustment(task, token);

        return false;
    }

    bool OptimizeExecutor::ExecuteGain(Task& task, CancellationToken /* token */)
    {
        if (!m_cudaPipeline || !m_nvJpegCodec) return false;

        int imageId = 0;
        try
        {
            imageId = std::stoi(task.unitKey);
        }
        catch (...)
        {
            return false;
        }

        auto refIdOpt = ReferenceImageId(m_projectManager);
        if (!refIdOpt.has_value()) return false;
        int refId = refIdOpt.value();

        if (imageId == refId)
        {
            // The reference image's own gain is always 1.0 by definition.
            return m_projectManager.UpdateImageGain(imageId, 1.0f);
        }

        auto pairTask = FindCompletedAlignPair(m_projectManager, refId, imageId);
        if (!pairTask.has_value())
        {
            // No successful alignment against the reference - 1.0 is a
            // valid, documented default (a successful result, not a
            // failure), matching v1's own "no gain data -> assume 1.0"
            // fallback intent.
            return m_projectManager.UpdateImageGain(imageId, 1.0f);
        }

        // Re-derive the correspondence set from the persisted feature
        // blobs rather than threading matched-point state through the
        // Task table - the same re-derivation approach bundle
        // adjustment (Task 4) also uses.
        auto refFeatureBlobId = FindAlignFeatureBlobId(m_projectManager, refId);
        auto targetFeatureBlobId = FindAlignFeatureBlobId(m_projectManager, imageId);
        if (!refFeatureBlobId.has_value() || !targetFeatureBlobId.has_value()) return false;

        auto refBytes = m_storageEngine.ReadBlob(refFeatureBlobId.value());
        auto targetBytes = m_storageEngine.ReadBlob(targetFeatureBlobId.value());
        if (!refBytes.has_value() || !targetBytes.has_value()) return false;

        std::vector<Compute::FeaturePoint> refPoints, targetPoints;
        std::vector<uint64_t> refDescWords, targetDescWords;
        if (!DeserializeFeatures(refBytes.value(), refPoints, refDescWords)) return false;
        if (!DeserializeFeatures(targetBytes.value(), targetPoints, targetDescWords)) return false;
        if (refPoints.empty() || targetPoints.empty())
        {
            return m_projectManager.UpdateImageGain(imageId, 1.0f);
        }

        std::vector<Compute::MatchResult> matches(kMaxMatchesForGain);
        int matchCount = 0;
        Compute::ComputeResult matchResult = m_cudaPipeline->MatchFeatures(
            reinterpret_cast<const Compute::BriefDescriptor*>(refDescWords.data()), static_cast<int>(refPoints.size()),
            reinterpret_cast<const Compute::BriefDescriptor*>(targetDescWords.data()), static_cast<int>(targetPoints.size()),
            matches.data(), &matchCount, kMaxMatchesForGain);
        if (matchResult != Compute::ComputeResult::SUCCESS || matchCount == 0)
        {
            return m_projectManager.UpdateImageGain(imageId, 1.0f);
        }

        const InputImageModel* refImage = nullptr;
        const InputImageModel* targetImage = nullptr;
        for (const auto& img : m_projectManager.GetInputImages())
        {
            if (img.id == refId) refImage = &img;
            if (img.id == imageId) targetImage = &img;
        }
        if (!refImage || !targetImage) return false;

        ImageLoader refLoader;
        ImageLoader targetLoader;
        if (!refLoader.Open(refImage->file_path) || !targetLoader.Open(targetImage->file_path)) return false;

        std::vector<unsigned char> refJpeg;
        std::vector<unsigned char> targetJpeg;
        if (!refLoader.GetEmbeddedPreviewJpeg(refJpeg) || !targetLoader.GetEmbeddedPreviewJpeg(targetJpeg)) return false;

        unsigned char* refRgb = nullptr;
        int refW = 0;
        int refH = 0;
        if (m_nvJpegCodec->Decode(refJpeg.data(), refJpeg.size(), &refRgb, &refW, &refH) != Compute::ComputeResult::SUCCESS)
            return false;

        unsigned char* targetRgb = nullptr;
        int targetW = 0;
        int targetH = 0;
        if (m_nvJpegCodec->Decode(targetJpeg.data(), targetJpeg.size(), &targetRgb, &targetW, &targetH) != Compute::ComputeResult::SUCCESS)
        {
            m_nvJpegCodec->FreeDecoded(refRgb);
            return false;
        }

        double refSum = 0.0;
        double targetSum = 0.0;
        int sampleCount = 0;

        for (int i = 0; i < matchCount; ++i)
        {
            const auto& match = matches[i];
            int rx = static_cast<int>(refPoints[match.indexA].x);
            int ry = static_cast<int>(refPoints[match.indexA].y);
            int tx = static_cast<int>(targetPoints[match.indexB].x);
            int ty = static_cast<int>(targetPoints[match.indexB].y);

            if (rx < 0 || ry < 0 || rx >= refW || ry >= refH) continue;
            if (tx < 0 || ty < 0 || tx >= targetW || ty >= targetH) continue;

            int refIdx = (ry * refW + rx) * 3;
            int targetIdx = (ty * targetW + tx) * 3;

            double refLuma = 0.299 * refRgb[refIdx] + 0.587 * refRgb[refIdx + 1] + 0.114 * refRgb[refIdx + 2];
            double targetLuma = 0.299 * targetRgb[targetIdx] + 0.587 * targetRgb[targetIdx + 1] + 0.114 * targetRgb[targetIdx + 2];

            refSum += refLuma;
            targetSum += targetLuma;
            ++sampleCount;
        }

        m_nvJpegCodec->FreeDecoded(refRgb);
        m_nvJpegCodec->FreeDecoded(targetRgb);

        if (sampleCount == 0 || targetSum <= 0.0)
        {
            return m_projectManager.UpdateImageGain(imageId, 1.0f);
        }

        float gain = static_cast<float>(refSum / targetSum);
        gain = std::clamp(gain, 0.25f, 4.0f);

        return m_projectManager.UpdateImageGain(imageId, gain);
    }

    bool OptimizeExecutor::ExecuteColorTransfer(Task& task, CancellationToken /* token */)
    {
        if (!m_cudaPipeline) return false;

        int imageId = 0;
        try
        {
            imageId = std::stoi(task.unitKey);
        }
        catch (...)
        {
            return false;
        }

        auto refIdOpt = ReferenceImageId(m_projectManager);
        if (!refIdOpt.has_value()) return false;
        int refId = refIdOpt.value();

        auto targetBlobId = FindIngestBlobId(m_projectManager, imageId);
        if (!targetBlobId.has_value()) return false;

        auto targetBuffer = m_storageEngine.ReadPixelBuffer(targetBlobId.value());
        if (!targetBuffer.has_value()) return false;

        if (imageId == refId)
        {
            // No-op passthrough for the reference image itself, so Render
            // can uniformly read "the color-transfer output blob" for
            // every image without a reference-image special case.
            auto blobId = m_storageEngine.WritePixelBuffer(targetBuffer.value(), "color_corrected_rgb48");
            if (!blobId.has_value()) return false;
            task.outputBlobId = blobId;
            return true;
        }

        auto refBlobId = FindIngestBlobId(m_projectManager, refId);
        if (!refBlobId.has_value()) return false;

        auto refBuffer = m_storageEngine.ReadPixelBuffer(refBlobId.value());
        if (!refBuffer.has_value()) return false;

        double refMean[3];
        double refStd[3];
        if (m_cudaPipeline->ComputeLabStats(refBuffer->data.data(), refBuffer->width, refBuffer->height, refMean, refStd)
            != Compute::ComputeResult::SUCCESS)
        {
            return false;
        }

        double targetMean[3];
        double targetStd[3];
        if (m_cudaPipeline->ComputeLabStats(targetBuffer->data.data(), targetBuffer->width, targetBuffer->height, targetMean, targetStd)
            != Compute::ComputeResult::SUCCESS)
        {
            return false;
        }

        PixelBuffer transferred = targetBuffer.value();
        if (m_cudaPipeline->ApplyReinhardColorTransfer(
                transferred.data.data(), transferred.width, transferred.height,
                targetMean, targetStd, refMean, refStd)
            != Compute::ComputeResult::SUCCESS)
        {
            return false;
        }

        auto blobId = m_storageEngine.WritePixelBuffer(transferred, "color_corrected_rgb48");
        if (!blobId.has_value()) return false;

        task.outputBlobId = blobId;
        return true;
    }

    // Implemented in a follow-up issue (Task 4).
    bool OptimizeExecutor::ExecuteBundleAdjustment(Task& /* task */, CancellationToken /* token */)
    {
        return false;
    }
}
