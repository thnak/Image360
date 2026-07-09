#include "pch.h"
#include "HeaderFiles/StorageEngine.h"
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace WindowsApp::Core
{
    StorageEngine::StorageEngine() = default;

    StorageEngine::~StorageEngine()
    {
        Close();
    }

    std::wstring StorageEngine::ShardPath(int shardIndex) const
    {
        std::wostringstream indexStream;
        indexStream << std::setw(4) << std::setfill(L'0') << shardIndex;
        return m_projectDirectory + L"\\" + m_projectBaseName + L"." + indexStream.str() + L".vfpdata";
    }

    bool StorageEngine::OpenOrCreateActiveShard()
    {
        int index = 1;
        while (PlatformFile::Exists(ShardPath(index)))
        {
            ++index;
        }
        // `index` is now the first non-existent shard number - the active
        // shard is the last existing one, or 1 if none exist yet.
        m_activeShardIndex = (index > 1) ? (index - 1) : 1;

        std::wstring path = ShardPath(m_activeShardIndex);
        if (!m_activeShardFile.Open(path, FileOpenMode::OpenOrCreate)) return false;

        m_activeShardOffset = m_activeShardFile.Size();
        return true;
    }

    bool StorageEngine::RollToNextShard()
    {
        m_activeShardFile.Close();

        ++m_activeShardIndex;
        std::wstring path = ShardPath(m_activeShardIndex);
        if (!m_activeShardFile.Open(path, FileOpenMode::CreateNew)) return false;

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
        m_activeShardFile.Close();
        m_projectManager = nullptr;
        m_projectDirectory.clear();
        m_projectBaseName.clear();
        m_activeShardIndex = 1;
        m_activeShardOffset = 0;
    }

    bool StorageEngine::IsOpen() const
    {
        return m_activeShardFile.IsOpen();
    }

    std::optional<int64_t> StorageEngine::WriteBlob(const void* data, size_t length, const std::string& formatTag)
    {
        std::lock_guard<std::mutex> lock(m_writeMutex);

        if (!m_activeShardFile.IsOpen() || !m_projectManager) return std::nullopt;

        if (m_activeShardOffset + length > kMaxShardBytes)
        {
            if (!RollToNextShard()) return std::nullopt;
        }

        if (!m_activeShardFile.SeekAbsolute(m_activeShardOffset)) return std::nullopt;
        if (!m_activeShardFile.Write(data, length)) return std::nullopt;

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
        PlatformFile file;
        if (!file.Open(path, FileOpenMode::ReadOnly)) return std::nullopt;

        if (!file.SeekAbsolute(static_cast<uint64_t>(entry->offset))) return std::nullopt;

        std::vector<uint8_t> buffer(static_cast<size_t>(entry->length));
        if (!buffer.empty() && !file.Read(buffer.data(), buffer.size())) return std::nullopt;

        return buffer;
    }

    std::optional<int64_t> StorageEngine::WritePixelBuffer(const PixelBuffer& buffer, const std::string& formatTag)
    {
        int32_t width = static_cast<int32_t>(buffer.width);
        int32_t height = static_cast<int32_t>(buffer.height);
        size_t pixelBytes = buffer.data.size() * sizeof(unsigned short);
        size_t totalBytes = sizeof(width) + sizeof(height) + pixelBytes;

        std::vector<uint8_t> framed(totalBytes);
        std::memcpy(framed.data(), &width, sizeof(width));
        std::memcpy(framed.data() + sizeof(width), &height, sizeof(height));
        if (pixelBytes > 0)
            std::memcpy(framed.data() + sizeof(width) + sizeof(height), buffer.data.data(), pixelBytes);

        return WriteBlob(framed.data(), framed.size(), formatTag);
    }

    std::optional<PixelBuffer> StorageEngine::ReadPixelBuffer(int64_t blobId)
    {
        auto bytes = ReadBlob(blobId);
        if (!bytes.has_value() || bytes->size() < sizeof(int32_t) * 2) return std::nullopt;

        int32_t width = 0;
        int32_t height = 0;
        std::memcpy(&width, bytes->data(), sizeof(width));
        std::memcpy(&height, bytes->data() + sizeof(width), sizeof(height));

        size_t headerSize = sizeof(width) + sizeof(height);
        size_t pixelBytes = bytes->size() - headerSize;
        if (pixelBytes % sizeof(unsigned short) != 0) return std::nullopt;

        PixelBuffer result;
        result.width = width;
        result.height = height;
        result.data.resize(pixelBytes / sizeof(unsigned short));
        if (pixelBytes > 0)
            std::memcpy(result.data.data(), bytes->data() + headerSize, pixelBytes);

        return result;
    }
}
