#include "pch.h"
#include "HeaderFiles/StorageEngine.h"
#include <algorithm>

namespace WindowsApp::Core
{
    namespace
    {
        constexpr DWORD kIoChunkBytes = 64u * 1024 * 1024; // 64 MB per WriteFile/ReadFile call
    }

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

    std::optional<int64_t> StorageEngine::WriteBlob(const void* data, size_t length, const std::string& formatTag)
    {
        if (m_activeShardHandle == INVALID_HANDLE_VALUE || !m_projectManager) return std::nullopt;

        if (m_activeShardOffset + length > kMaxShardBytes)
        {
            if (!RollToNextShard()) return std::nullopt;
        }

        LARGE_INTEGER seekPos;
        seekPos.QuadPart = static_cast<LONGLONG>(m_activeShardOffset);
        if (!SetFilePointerEx(m_activeShardHandle, seekPos, nullptr, FILE_BEGIN)) return std::nullopt;

        const uint8_t* bytes = static_cast<const uint8_t*>(data);
        size_t totalWritten = 0;
        while (totalWritten < length)
        {
            DWORD toWrite = static_cast<DWORD>(std::min<size_t>(length - totalWritten, kIoChunkBytes));
            DWORD written = 0;
            if (!WriteFile(m_activeShardHandle, bytes + totalWritten, toWrite, &written, nullptr) || written == 0)
                return std::nullopt;
            totalWritten += written;
        }

        std::wstring fullShardPath = ShardPath(m_activeShardIndex);
        size_t lastSlash = fullShardPath.find_last_of(L'\\');

        BlobDirectoryEntry entry;
        entry.shardFile = (lastSlash == std::wstring::npos) ? fullShardPath : fullShardPath.substr(lastSlash + 1);
        entry.offset = static_cast<int64_t>(m_activeShardOffset);
        entry.length = static_cast<int64_t>(length);
        entry.compressedLength = std::nullopt;
        entry.formatTag = formatTag;

        int64_t blobId = m_projectManager->AddBlobDirectoryEntry(entry);
        if (blobId <= 0) return std::nullopt;

        m_activeShardOffset += length;
        return blobId;
    }

    std::optional<std::vector<uint8_t>> StorageEngine::ReadBlob(int64_t blobId)
    {
        if (!m_projectManager) return std::nullopt;

        auto entry = m_projectManager->GetBlobDirectoryEntry(blobId);
        if (!entry.has_value()) return std::nullopt;

        std::wstring path = m_projectDirectory + L"\\" + entry->shardFile;
        HANDLE handle = CreateFileW(
            path.c_str(), GENERIC_READ, FILE_SHARE_READ,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle == INVALID_HANDLE_VALUE) return std::nullopt;

        LARGE_INTEGER seekPos;
        seekPos.QuadPart = entry->offset;
        if (!SetFilePointerEx(handle, seekPos, nullptr, FILE_BEGIN))
        {
            CloseHandle(handle);
            return std::nullopt;
        }

        std::vector<uint8_t> buffer(static_cast<size_t>(entry->length));
        size_t totalRead = 0;
        while (totalRead < buffer.size())
        {
            DWORD toRead = static_cast<DWORD>(std::min<size_t>(buffer.size() - totalRead, kIoChunkBytes));
            DWORD readBytes = 0;
            if (!ReadFile(handle, buffer.data() + totalRead, toRead, &readBytes, nullptr) || readBytes == 0)
            {
                CloseHandle(handle);
                return std::nullopt;
            }
            totalRead += readBytes;
        }

        CloseHandle(handle);
        return buffer;
    }
}
