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
    // VRAM-budget-driven chunk sizing (docs/ARCHITECTURE.md SS6, SS7.4).
    // Free function, no Compute::GpuInfo/CUDA dependency - callers that
    // have a Compute::GpuInfo (the project-creation UI, PipelineDriver
    // setup) pass gpuInfo.totalMemory here before calling CreateProject.
    // Thresholds match ARCHITECTURE.md SS7.4's example numbers exactly.
    int RecommendedChunkSize(uint64_t totalVramBytes);

    // Same idea for the CPU backend, but a distinct function rather than
    // an overload with different-meaning numbers - "chunk size vs. system
    // RAM" is a different curve than "chunk size vs. VRAM": RAM is
    // typically larger, but shared with the OS/other apps and with
    // TaskScheduler's own in-flight task window (each in-flight
    // Render/Align/Optimize task holds a full chunk-sized buffer), so
    // thresholds are conservative relative to raw RAM size. Callers with a
    // CpuComputeBackend pass GetGpuInfo().totalMemory (populated with
    // system RAM for that backend) here instead of VRAM.
    int RecommendedChunkSizeForCpu(uint64_t totalRamBytes);

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
        bool CreateProject(const std::wstring& dbPath, int totalWidth, int totalHeight, int chunkSize = 4096);

        // Burst-mode entry point (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS3, SS8
        // Phase 0) - deliberately separate from CreateProject rather than an
        // overload, since a burst project has no chunk grid (one merged
        // output frame, not a tiled canvas): totalWidth/totalHeight/
        // chunkSize would be meaningless parameters for it. GetTotalWidth/
        // GetTotalHeight report 0 until a real burst executor sets them
        // from the first ingested frame - out of scope here.
        bool CreateBurstProject(const std::wstring& dbPath, BurstMode mode);

        bool LoadProject(const std::wstring& dbPath);
        void CloseProject();

        // A freshly-constructed ProjectManager (no project loaded yet) and
        // every project created before this existed report PANORAMA/NONE -
        // the project_metadata table simply has no project_type/burst_mode
        // key, so no .vfp migration was needed to add these.
        ProjectType GetProjectType() const { return m_projectType; }
        BurstMode GetBurstMode() const { return m_burstMode; }

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

        // Burst-mode task seeding (docs/COMPUTATIONAL_PHOTOGRAPHY.md SS3,
        // docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md).
        // One BURST_ALIGN/"frame" task per input image (a burst project's
        // "frames" are stored via the same AddInputImage/input_images path
        // panorama's input images use). Call after every frame has been
        // added.
        bool SeedBurstAlignTasks();

        // Exactly one BURST_MERGE/"output"/"merged" task and one
        // BURST_FINISH/"output"/"final" task - safe to seed upfront
        // alongside SeedBurstAlignTasks, unlike panorama's SeedRenderTasks
        // (see this method's own .cpp comment for why).
        bool SeedBurstMergeTasks();

        // One STAGE1_ALIGN/"image" task per input image (feature
        // extraction); one STAGE1_ALIGN/"pair" task per (i, j) with i < j
        // over all input images (match + RANSAC) - all-pairs, an
        // explicit O(n^2) scope cut documented in
        // docs/superpowers/plans/2026-07-07-align-stage.md's Global
        // Constraints.
        bool SeedAlignTasks();

        // One STAGE2_OPTIMIZE/"gain" + one STAGE2_OPTIMIZE/"color" task
        // per input image, exactly one STAGE2_OPTIMIZE/"ba_checkpoint"
        // task (unit_key = "global").
        bool SeedOptimizeTasks();

        // Runs overlap culling (OverlapCulling.h) for every chunk,
        // populates chunk_contributors, and seeds one
        // STAGE3_RENDER/"chunk" task per chunk with at least one
        // contributor - chunks with zero contributors get no task at
        // all (unlike v1's ProcessChunk, which still dispatched and
        // just logged "no images overlap"). Must be called after
        // Optimize completes (needs final homographies) - PipelineDriver
        // wires this timing, not this method itself.
        bool SeedRenderTasks();
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
        ProjectType m_projectType = ProjectType::PANORAMA;
        BurstMode m_burstMode = BurstMode::NONE;
        std::vector<ChunkModel> m_chunks;
        std::vector<InputImageModel> m_inputImages;

        bool ExecuteNonQuery(const std::string& sql);
        // Shared by CreateProject/CreateBurstProject: opens the DB, sets
        // WAL/synchronous pragmas, creates every table both project types
        // need (chunks/chunk_contributors stay unused/empty for a burst
        // project, same as tasks/blob_directory stay unused/empty for a
        // never-run panorama project - creating all tables unconditionally
        // is simpler than a schema that varies by ProjectType). Returns
        // false (and leaves m_db closed) on any failure, matching
        // CreateProject's existing error contract.
        bool OpenAndCreateSchema(const std::wstring& dbPath);
        void LoadMetadata();
        void LoadChunks();
        void LoadInputImages();
    };
}
