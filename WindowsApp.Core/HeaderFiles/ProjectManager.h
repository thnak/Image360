#pragma once
#include "Types.h"
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <cstdint>

struct sqlite3;

namespace WindowsApp::Core
{
    struct InputImageModel
    {
        int id = 0;
        std::wstring file_path;
        Homography homography;
        float gain = 1.0f;
        CfaType cfaType = CfaType::UNKNOWN;
    };

    class ProjectManager
    {
    public:
        ProjectManager();
        ~ProjectManager();

        // Database operations
        bool CreateProject(const std::wstring& dbPath, int totalWidth, int totalHeight);
        bool LoadProject(const std::wstring& dbPath);
        void CloseProject();

        bool AddInputImage(const std::wstring& filePath, const Homography& h, CfaType cfaType = CfaType::BAYER);
        bool UpdateImageGain(int imageId, float gain);

        // Overwrites a previously-computed homography - AddInputImage
        // only ever sets the initial identity one; Align is the first
        // stage that computes a real one.
        bool UpdateHomography(int imageId, const Homography& h);

        // Creates one PipelineStage::STAGE0_INGEST / unit_kind="image"
        // task per input image row that doesn't already have one.
        // Idempotent via CreateTasksIfAbsent's UNIQUE(stage, unit_kind,
        // unit_key) constraint.
        bool SeedIngestTasks();

        // One STAGE1_ALIGN/"image" task per input image (feature
        // extraction); one STAGE1_ALIGN/"pair" task per (i, j) with i < j
        // over all input images (match + RANSAC) - all-pairs, an
        // explicit O(n^2) scope cut documented in
        // docs/superpowers/plans/2026-07-07-align-stage.md's Global
        // Constraints.
        bool SeedAlignTasks();
        bool UpdateChunkStatus(const std::string& chunkId, ChunkStatus status, const std::wstring& cachePath);

        // Tasks
        bool CreateTasksIfAbsent(const std::vector<Task>& tasks);
        std::vector<Task> GetTasksForStage(PipelineStage stage) const;
        bool UpdateTaskStatus(int64_t taskId, TaskStatus status);
        bool CommitTaskOutput(int64_t taskId, int64_t outputBlobId);
        bool UpdateTaskCheckpoint(int64_t taskId, const std::string& checkpointJson);
        int  ReclaimStaleRunningTasks(PipelineStage stage);

        // Chunk contributors
        bool SetChunkContributors(const std::string& chunkId, const std::vector<int>& imageIds);
        std::vector<int> GetChunkContributors(const std::string& chunkId) const;

        // Blob directory
        int64_t AddBlobDirectoryEntry(const BlobDirectoryEntry& entry);
        std::optional<BlobDirectoryEntry> GetBlobDirectoryEntry(int64_t blobId) const;

        // Accessors
        const std::vector<ChunkModel>& GetChunks() const { return m_chunks; }
        const std::vector<InputImageModel>& GetInputImages() const { return m_inputImages; }
        
        int GetTotalWidth() const { return m_totalWidth; }
        int GetTotalHeight() const { return m_totalHeight; }
        std::wstring GetProjectPath() const { return m_projectPath; }

    private:
        sqlite3* m_db = nullptr;
        std::wstring m_projectPath;
        int m_totalWidth = 0;
        int m_totalHeight = 0;
        std::vector<ChunkModel> m_chunks;
        std::vector<InputImageModel> m_inputImages;

        bool ExecuteNonQuery(const std::string& sql);
        void LoadMetadata();
        void LoadChunks();
        void LoadInputImages();
    };
}
