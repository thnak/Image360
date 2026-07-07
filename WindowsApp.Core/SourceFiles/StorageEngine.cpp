#include "pch.h"
#include "HeaderFiles/StorageEngine.h"

namespace WindowsApp::Core
{
    StorageEngine::StorageEngine() = default;

    StorageEngine::~StorageEngine()
    {
        Close();
    }

    std::wstring StorageEngine::ShardPath(int shardIndex) const
    {
        wchar_t indexBuffer[16];
        swprintf_s(indexBuffer, L"%04d", shardIndex);
        return m_projectDirectory + L"\\" + m_projectBaseName + L"." + indexBuffer + L".vfpdata";
    }

    bool StorageEngine::OpenOrCreateActiveShard()
    {
        int index = 1;
        while (GetFileAttributesW(ShardPath(index).c_str()) != INVALID_FILE_ATTRIBUTES)
        {
            ++index;
        }
        // `index` is now the first non-existent shard number - the active
        // shard is the last existing one, or 1 if none exist yet.
        m_activeShardIndex = (index > 1) ? (index - 1) : 1;

        std::wstring path = ShardPath(m_activeShardIndex);
        m_activeShardHandle = CreateFileW(
            path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
            nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (m_activeShardHandle == INVALID_HANDLE_VALUE) return false;

        LARGE_INTEGER size{};
        if (!GetFileSizeEx(m_activeShardHandle, &size))
        {
            CloseHandle(m_activeShardHandle);
            m_activeShardHandle = INVALID_HANDLE_VALUE;
            return false;
        }
        m_activeShardOffset = static_cast<uint64_t>(size.QuadPart);
        return true;
    }

    bool StorageEngine::RollToNextShard()
    {
        if (m_activeShardHandle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(m_activeShardHandle);
            m_activeShardHandle = INVALID_HANDLE_VALUE;
        }

        ++m_activeShardIndex;
        std::wstring path = ShardPath(m_activeShardIndex);
        m_activeShardHandle = CreateFileW(
            path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (m_activeShardHandle == INVALID_HANDLE_VALUE) return false;

        m_activeShardOffset = 0;
        return true;
    }

    bool StorageEngine::Open(const std::wstring& projectDirectory,
                              const std::wstring& projectBaseName,
                              ProjectManager& projectManager)
    {
        Close();
        m_projectDirectory = projectDirectory;
        m_projectBaseName = projectBaseName;
        m_projectManager = &projectManager;
        return OpenOrCreateActiveShard();
    }

    void StorageEngine::Close()
    {
        if (m_activeShardHandle != INVALID_HANDLE_VALUE)
        {
            CloseHandle(m_activeShardHandle);
            m_activeShardHandle = INVALID_HANDLE_VALUE;
        }
        m_projectManager = nullptr;
        m_projectDirectory.clear();
        m_projectBaseName.clear();
        m_activeShardIndex = 1;
        m_activeShardOffset = 0;
    }

    bool StorageEngine::IsOpen() const
    {
        return m_activeShardHandle != INVALID_HANDLE_VALUE;
    }
}
