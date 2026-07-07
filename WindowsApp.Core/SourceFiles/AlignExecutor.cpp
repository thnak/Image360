#include "pch.h"
#include "HeaderFiles/AlignExecutor.h"
#include "HeaderFiles/ImageLoader.h"
#include "HeaderFiles/RansacHomography.h"
#include <cstring>
#include <string>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        constexpr int kMaxFeatures = 2000;
        constexpr int kMaxMatches = 2000;

        // Feature blob layout: [int32 count][count * FeaturePoint][count * 4 uint64_t descriptor words].
        // BriefDescriptor (uint64_t[4]) is an array type - not
        // CopyAssignable, so it can't live in a std::vector directly.
        // Descriptors are stored/handled as a flat std::vector<uint64_t>
        // (4 words per descriptor) instead, reinterpret_cast to
        // Compute::BriefDescriptor* only at CudaPipeline call sites.
        std::vector<uint8_t> SerializeFeatures(
            const std::vector<Compute::FeaturePoint>& points,
            const std::vector<uint64_t>& descriptorWords)
        {
            int32_t count = static_cast<int32_t>(points.size());
            size_t total = sizeof(count)
                + points.size() * sizeof(Compute::FeaturePoint)
                + descriptorWords.size() * sizeof(uint64_t);

            std::vector<uint8_t> buffer(total);
            size_t offset = 0;
            std::memcpy(buffer.data() + offset, &count, sizeof(count));
            offset += sizeof(count);

            if (!points.empty())
            {
                std::memcpy(buffer.data() + offset, points.data(), points.size() * sizeof(Compute::FeaturePoint));
                offset += points.size() * sizeof(Compute::FeaturePoint);
            }
            if (!descriptorWords.empty())
            {
                std::memcpy(buffer.data() + offset, descriptorWords.data(), descriptorWords.size() * sizeof(uint64_t));
            }
            return buffer;
        }

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

        bool ParseImagePair(const std::string& unitKey, int& imageA, int& imageB)
        {
            size_t colon = unitKey.find(':');
            if (colon == std::string::npos) return false;

            try
            {
                imageA = std::stoi(unitKey.substr(0, colon));
                imageB = std::stoi(unitKey.substr(colon + 1));
            }
            catch (...)
            {
                return false;
            }
            return true;
        }

        // Finds the STAGE1_ALIGN/"image" task for `imageId` and returns
        // its output_blob_id if COMPLETED.
        std::optional<int64_t> FindFeatureBlobId(ProjectManager& projectManager, int imageId)
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
    }

    AlignExecutor::AlignExecutor(ProjectManager& projectManager, StorageEngine& storageEngine,
                                  std::shared_ptr<Compute::CudaPipeline> cudaPipeline,
                                  std::shared_ptr<Compute::NvJpegCodec> nvJpegCodec)
        : m_projectManager(projectManager)
        , m_storageEngine(storageEngine)
        , m_cudaPipeline(std::move(cudaPipeline))
        , m_nvJpegCodec(std::move(nvJpegCodec))
    {
    }

    bool AlignExecutor::Execute(Task& task, CancellationToken token)
    {
        if (token.stop_requested()) return false;

        if (task.unitKind == "image") return ExecuteFeatureExtraction(task, token);
        if (task.unitKind == "pair") return ExecuteMatch(task, token);

        // Unknown unit_kind is a data/programmer error (e.g. a hand-
        // edited DB), not a crash-worthy bug - still an expected failure.
        return false;
    }

    bool AlignExecutor::ExecuteFeatureExtraction(Task& task, CancellationToken /* token */)
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

        const InputImageModel* image = nullptr;
        for (const auto& img : m_projectManager.GetInputImages())
        {
            if (img.id == imageId)
            {
                image = &img;
                break;
            }
        }
        if (!image) return false;

        ImageLoader loader;
        if (!loader.Open(image->file_path)) return false;

        std::vector<unsigned char> jpegBytes;
        if (!loader.GetEmbeddedPreviewJpeg(jpegBytes)) return false;

        unsigned char* decodedRgb = nullptr;
        int width = 0;
        int height = 0;
        Compute::ComputeResult decodeResult = m_nvJpegCodec->Decode(
            jpegBytes.data(), jpegBytes.size(), &decodedRgb, &width, &height);
        if (decodeResult != Compute::ComputeResult::SUCCESS) return false;

        std::vector<Compute::FeaturePoint> points(kMaxFeatures);
        std::vector<uint64_t> descriptorWords(static_cast<size_t>(kMaxFeatures) * 4);
        int detectedCount = 0;

        Compute::ComputeResult featureResult = m_cudaPipeline->DetectAndDescribeFeatures(
            decodedRgb, width, height,
            points.data(), reinterpret_cast<Compute::BriefDescriptor*>(descriptorWords.data()),
            &detectedCount, kMaxFeatures);

        m_nvJpegCodec->FreeDecoded(decodedRgb);

        if (featureResult != Compute::ComputeResult::SUCCESS) return false;

        points.resize(detectedCount);
        descriptorWords.resize(static_cast<size_t>(detectedCount) * 4);

        std::vector<uint8_t> blobBytes = SerializeFeatures(points, descriptorWords);

        auto blobId = m_storageEngine.WriteBlob(blobBytes.data(), blobBytes.size(), "features_brief256");
        if (!blobId.has_value()) return false;

        task.outputBlobId = blobId;
        return true;
    }

    bool AlignExecutor::ExecuteMatch(Task& task, CancellationToken /* token */)
    {
        if (!m_cudaPipeline) return false;

        int imageA = 0;
        int imageB = 0;
        if (!ParseImagePair(task.unitKey, imageA, imageB)) return false;

        auto blobIdA = FindFeatureBlobId(m_projectManager, imageA);
        auto blobIdB = FindFeatureBlobId(m_projectManager, imageB);
        if (!blobIdA.has_value() || !blobIdB.has_value()) return false;

        auto bytesA = m_storageEngine.ReadBlob(blobIdA.value());
        auto bytesB = m_storageEngine.ReadBlob(blobIdB.value());
        if (!bytesA.has_value() || !bytesB.has_value()) return false;

        std::vector<Compute::FeaturePoint> pointsA, pointsB;
        std::vector<uint64_t> descWordsA, descWordsB;
        if (!DeserializeFeatures(bytesA.value(), pointsA, descWordsA)) return false;
        if (!DeserializeFeatures(bytesB.value(), pointsB, descWordsB)) return false;

        if (pointsA.empty() || pointsB.empty()) return false;

        std::vector<Compute::MatchResult> matches(kMaxMatches);
        int matchCount = 0;
        Compute::ComputeResult matchResult = m_cudaPipeline->MatchFeatures(
            reinterpret_cast<const Compute::BriefDescriptor*>(descWordsA.data()), static_cast<int>(pointsA.size()),
            reinterpret_cast<const Compute::BriefDescriptor*>(descWordsB.data()), static_cast<int>(pointsB.size()),
            matches.data(), &matchCount, kMaxMatches);
        if (matchResult != Compute::ComputeResult::SUCCESS) return false;
        if (matchCount < 4) return false; // not enough correspondences for a homography

        std::vector<std::pair<Compute::FeaturePoint, Compute::FeaturePoint>> correspondences;
        correspondences.reserve(matchCount);
        for (int i = 0; i < matchCount; ++i)
        {
            correspondences.emplace_back(pointsA[matches[i].indexA], pointsB[matches[i].indexB]);
        }

        RansacResult ransacResult = RunRansacHomography(*m_cudaPipeline, correspondences);
        if (!ransacResult.success) return false;

        // Image A stays the reference frame in this v1 pass - a
        // documented simplification (docs/superpowers/plans/
        // 2026-07-07-align-stage.md Task 6 Step 3); proper bundle-
        // adjustment-driven global alignment is Optimize's job.
        if (!m_projectManager.UpdateHomography(imageB, ransacResult.homography)) return false;

        return true;
    }
}
