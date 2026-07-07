#pragma once
#include "ProjectManager.h"
#include <string>
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
