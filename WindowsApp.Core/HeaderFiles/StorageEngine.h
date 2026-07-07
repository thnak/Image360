#pragma once
#include "ProjectManager.h"
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <Windows.h>

namespace WindowsApp::Core
{
    // Sharded <project>.NNNN.vfpdata blob container (docs/ARCHITECTURE.md
    // SS5) - WriteFile-based path only, DirectStorage is a later plan (see
    // docs/superpowers/plans/2026-07-07-storage-engine.md's scope cut).
    //
    // DURABILITY ORDERING: callers must call
    // ProjectManager::CommitTaskOutput(taskId, blobId) only AFTER WriteBlob
    // returns successfully - never before. See WriteBlob's own comment.
    class StorageEngine
    {
    public:
        StorageEngine();
        ~StorageEngine();
        StorageEngine(const StorageEngine&) = delete;
        StorageEngine& operator=(const StorageEngine&) = delete;

        // projectDirectory: folder containing the .vfp file.
        // projectBaseName: the .vfp file's stem (no extension) - shard
        // files are named "<projectBaseName>.NNNN.vfpdata" in that folder.
        bool Open(const std::wstring& projectDirectory,
                  const std::wstring& projectBaseName,
                  ProjectManager& projectManager);
        void Close();
        bool IsOpen() const;

        // Appends `length` bytes from `data` to the active shard (rolling
        // to a new shard first if this write would exceed kMaxShardBytes),
        // registers a blob_directory row via
        // ProjectManager::AddBlobDirectoryEntry, and returns the new
        // blobId.
        //
        // DURABILITY ORDERING (docs/ARCHITECTURE.md SS5, SS7.2): the
        // caller MUST call
        // ProjectManager::CommitTaskOutput(taskId, blobId) only AFTER
        // this returns successfully - never before, never speculatively.
        // If the process dies between this call and CommitTaskOutput,
        // the written bytes are simply an orphaned, harmless blob and
        // the owning task re-runs from scratch on resume (idempotency,
        // SS7.1). The reverse ordering would mark a task COMPLETED with
        // no data backing it.
        std::optional<int64_t> WriteBlob(const void* data, size_t length, const std::string& formatTag);

        // Looks up the blob's BlobDirectoryEntry, opens its shard (which
        // may not be the currently-active one), and reads back the full
        // byte range.
        std::optional<std::vector<uint8_t>> ReadBlob(int64_t blobId);

        // Convenience wrappers for the common case of storing a decoded/
        // rendered image. Framing: a fixed 8-byte header (int32 width,
        // int32 height) immediately followed by the raw pixel data
        // (matches PixelBuffer::data's layout) - blob_directory itself
        // has no width/height columns, so the dimensions travel inside
        // the blob payload instead of widening that table for one
        // consumer.
        std::optional<int64_t> WritePixelBuffer(const PixelBuffer& buffer, const std::string& formatTag);
        std::optional<PixelBuffer> ReadPixelBuffer(int64_t blobId);

    private:
        ProjectManager* m_projectManager = nullptr;
        std::wstring m_projectDirectory;
        std::wstring m_projectBaseName;
        int m_activeShardIndex = 1;
        HANDLE m_activeShardHandle = INVALID_HANDLE_VALUE;
        uint64_t m_activeShardOffset = 0;

        static constexpr uint64_t kMaxShardBytes = 4ull * 1024 * 1024 * 1024; // 4 GB

        std::wstring ShardPath(int shardIndex) const;
        bool OpenOrCreateActiveShard(); // finds highest existing NNNN or starts at 0001
        bool RollToNextShard();
    };
}
