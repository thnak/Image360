#pragma once
#include "Types.h"
#include <string>
#include <vector>
#include <memory>

struct sqlite3;

namespace WindowsApp::Core
{
    struct InputImageModel
    {
        int id = 0;
        std::wstring file_path;
        Homography homography;
        float gain = 1.0f;
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

        bool AddInputImage(const std::wstring& filePath, const Homography& h);
        bool UpdateImageGain(int imageId, float gain);
        bool UpdateChunkStatus(const std::string& chunkId, ChunkStatus status, const std::wstring& cachePath);

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
