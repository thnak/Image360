#include "pch.h"
#include "HeaderFiles/ProjectManager.h"
#include "sqlite3/sqlite3.h"
#include <stdexcept>

namespace WindowsApp::Core
{
    ProjectManager::ProjectManager() = default;

    ProjectManager::~ProjectManager()
    {
        CloseProject();
    }

    bool ProjectManager::CreateProject(const std::wstring& dbPath, int totalWidth, int totalHeight)
    {
        CloseProject();

        // Convert wide path to UTF-8 for SQLite
        int len = WideCharToMultiByte(CP_UTF8, 0, dbPath.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8Path(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, dbPath.c_str(), -1, utf8Path.data(), len, nullptr, nullptr);

        int rc = sqlite3_open(utf8Path.c_str(), &m_db);
        if (rc != SQLITE_OK)
        {
            m_db = nullptr;
            return false;
        }

        // Enable WAL mode for better concurrent read/write performance
        ExecuteNonQuery("PRAGMA journal_mode=WAL;");
        ExecuteNonQuery("PRAGMA synchronous=NORMAL;");

        // Create schema
        const char* schema = R"(
            CREATE TABLE IF NOT EXISTS project_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS input_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                h00 REAL DEFAULT 1.0, h01 REAL DEFAULT 0.0, h02 REAL DEFAULT 0.0,
                h10 REAL DEFAULT 0.0, h11 REAL DEFAULT 1.0, h12 REAL DEFAULT 0.0,
                h20 REAL DEFAULT 0.0, h21 REAL DEFAULT 0.0, h22 REAL DEFAULT 1.0,
                gain REAL DEFAULT 1.0
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                x_offset INTEGER NOT NULL,
                y_offset INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                cache_path TEXT
            );

            CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                unit_kind TEXT NOT NULL,
                unit_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                output_blob_id INTEGER,
                checkpoint_json TEXT,
                updated_at INTEGER NOT NULL DEFAULT 0,
                UNIQUE(stage, unit_kind, unit_key)
            );

            CREATE TABLE IF NOT EXISTS chunk_contributors (
                chunk_id TEXT NOT NULL,
                image_id INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, image_id)
            );

            CREATE TABLE IF NOT EXISTS blob_directory (
                blob_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shard_file TEXT NOT NULL,
                offset INTEGER NOT NULL,
                length INTEGER NOT NULL,
                compressed_length INTEGER,
                format_tag TEXT NOT NULL
            );
        )";

        if (!ExecuteNonQuery(schema))
        {
            CloseProject();
            return false;
        }

        // Store metadata
        std::string sql = "INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('total_width', '"
            + std::to_string(totalWidth) + "');";
        ExecuteNonQuery(sql);

        sql = "INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('total_height', '"
            + std::to_string(totalHeight) + "');";
        ExecuteNonQuery(sql);

        m_projectPath = dbPath;
        m_totalWidth = totalWidth;
        m_totalHeight = totalHeight;
        m_chunks.clear();
        m_inputImages.clear();

        // Generate chunk grid (4096x4096 tiles)
        const int chunkSize = 4096;
        for (int y = 0; y < totalHeight; y += chunkSize)
        {
            for (int x = 0; x < totalWidth; x += chunkSize)
            {
                int w = std::min(chunkSize, totalWidth - x);
                int h = std::min(chunkSize, totalHeight - y);

                ChunkModel chunk;
                chunk.id = "C_" + std::to_string(x / chunkSize) + "_" + std::to_string(y / chunkSize);
                chunk.x_offset = x;
                chunk.y_offset = y;
                chunk.width = w;
                chunk.height = h;
                chunk.status = ChunkStatus::PENDING;

                sql = "INSERT INTO chunks (id, x_offset, y_offset, width, height, status) VALUES ('"
                    + chunk.id + "', " + std::to_string(x) + ", " + std::to_string(y)
                    + ", " + std::to_string(w) + ", " + std::to_string(h) + ", 'PENDING');";
                ExecuteNonQuery(sql);

                m_chunks.push_back(std::move(chunk));
            }
        }

        return true;
    }

    bool ProjectManager::LoadProject(const std::wstring& dbPath)
    {
        CloseProject();

        int len = WideCharToMultiByte(CP_UTF8, 0, dbPath.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8Path(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, dbPath.c_str(), -1, utf8Path.data(), len, nullptr, nullptr);

        int rc = sqlite3_open(utf8Path.c_str(), &m_db);
        if (rc != SQLITE_OK)
        {
            m_db = nullptr;
            return false;
        }

        m_projectPath = dbPath;

        // Ensure tables added after this project was first created exist too,
        // so opening an older .vfp file never fails on a missing table.
        const char* taskSchema = R"(
            CREATE TABLE IF NOT EXISTS tasks (
                task_id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL,
                unit_kind TEXT NOT NULL,
                unit_key TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'PENDING',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                output_blob_id INTEGER,
                checkpoint_json TEXT,
                updated_at INTEGER NOT NULL DEFAULT 0,
                UNIQUE(stage, unit_kind, unit_key)
            );

            CREATE TABLE IF NOT EXISTS chunk_contributors (
                chunk_id TEXT NOT NULL,
                image_id INTEGER NOT NULL,
                PRIMARY KEY (chunk_id, image_id)
            );

            CREATE TABLE IF NOT EXISTS blob_directory (
                blob_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shard_file TEXT NOT NULL,
                offset INTEGER NOT NULL,
                length INTEGER NOT NULL,
                compressed_length INTEGER,
                format_tag TEXT NOT NULL
            );
        )";
        ExecuteNonQuery(taskSchema);

        LoadMetadata();
        LoadInputImages();
        LoadChunks();

        return true;
    }

    void ProjectManager::CloseProject()
    {
        if (m_db)
        {
            sqlite3_close(m_db);
            m_db = nullptr;
        }
        m_projectPath.clear();
        m_totalWidth = 0;
        m_totalHeight = 0;
        m_chunks.clear();
        m_inputImages.clear();
    }

    bool ProjectManager::AddInputImage(const std::wstring& filePath, const Homography& h)
    {
        if (!m_db) return false;

        int len = WideCharToMultiByte(CP_UTF8, 0, filePath.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8Path(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, filePath.c_str(), -1, utf8Path.data(), len, nullptr, nullptr);

        char sql[1024];
        snprintf(sql, sizeof(sql),
            "INSERT INTO input_images (file_path, h00, h01, h02, h10, h11, h12, h20, h21, h22) "
            "VALUES ('%s', %f, %f, %f, %f, %f, %f, %f, %f, %f);",
            utf8Path.c_str(),
            h.h[0], h.h[1], h.h[2],
            h.h[3], h.h[4], h.h[5],
            h.h[6], h.h[7], h.h[8]);

        if (!ExecuteNonQuery(sql)) return false;

        // Reload to get the auto-generated ID
        LoadInputImages();
        return true;
    }

    bool ProjectManager::UpdateImageGain(int imageId, float gain)
    {
        if (!m_db) return false;

        char sql[256];
        snprintf(sql, sizeof(sql),
            "UPDATE input_images SET gain = %f WHERE id = %d;", gain, imageId);

        if (!ExecuteNonQuery(sql)) return false;

        // Update local cache
        for (auto& img : m_inputImages)
        {
            if (img.id == imageId)
            {
                img.gain = gain;
                break;
            }
        }
        return true;
    }

    bool ProjectManager::UpdateChunkStatus(const std::string& chunkId, ChunkStatus status, const std::wstring& cachePath)
    {
        if (!m_db) return false;

        const char* statusStr = "PENDING";
        switch (status)
        {
        case ChunkStatus::PENDING:    statusStr = "PENDING"; break;
        case ChunkStatus::PROCESSING: statusStr = "PROCESSING"; break;
        case ChunkStatus::COMPLETED:  statusStr = "COMPLETED"; break;
        case ChunkStatus::FAILED:     statusStr = "FAILED"; break;
        }

        int len = WideCharToMultiByte(CP_UTF8, 0, cachePath.c_str(), -1, nullptr, 0, nullptr, nullptr);
        std::string utf8Cache(len - 1, '\0');
        WideCharToMultiByte(CP_UTF8, 0, cachePath.c_str(), -1, utf8Cache.data(), len, nullptr, nullptr);

        char sql[1024];
        snprintf(sql, sizeof(sql),
            "UPDATE chunks SET status = '%s', cache_path = '%s' WHERE id = '%s';",
            statusStr, utf8Cache.c_str(), chunkId.c_str());

        if (!ExecuteNonQuery(sql)) return false;

        // Update local cache
        for (auto& chunk : m_chunks)
        {
            if (chunk.id == chunkId)
            {
                chunk.status = status;
                chunk.cache_path = cachePath;
                break;
            }
        }
        return true;
    }

    bool ProjectManager::ExecuteNonQuery(const std::string& sql)
    {
        if (!m_db) return false;

        char* errMsg = nullptr;
        int rc = sqlite3_exec(m_db, sql.c_str(), nullptr, nullptr, &errMsg);
        if (rc != SQLITE_OK)
        {
            if (errMsg) sqlite3_free(errMsg);
            return false;
        }
        return true;
    }

    void ProjectManager::LoadMetadata()
    {
        if (!m_db) return;

        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT key, value FROM project_metadata;";
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) == SQLITE_OK)
        {
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                const char* key = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                const char* value = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));

                if (key && value)
                {
                    std::string k(key);
                    if (k == "total_width") m_totalWidth = std::stoi(value);
                    else if (k == "total_height") m_totalHeight = std::stoi(value);
                }
            }
        }
        sqlite3_finalize(stmt);
    }

    void ProjectManager::LoadChunks()
    {
        m_chunks.clear();
        if (!m_db) return;

        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT id, x_offset, y_offset, width, height, status, cache_path FROM chunks ORDER BY id;";
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) == SQLITE_OK)
        {
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                ChunkModel chunk;
                chunk.id = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
                chunk.x_offset = sqlite3_column_int(stmt, 1);
                chunk.y_offset = sqlite3_column_int(stmt, 2);
                chunk.width = sqlite3_column_int(stmt, 3);
                chunk.height = sqlite3_column_int(stmt, 4);

                std::string status = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
                if (status == "PENDING") chunk.status = ChunkStatus::PENDING;
                else if (status == "PROCESSING") chunk.status = ChunkStatus::PROCESSING;
                else if (status == "COMPLETED") chunk.status = ChunkStatus::COMPLETED;
                else if (status == "FAILED") chunk.status = ChunkStatus::FAILED;

                const wchar_t* cachePath = reinterpret_cast<const wchar_t*>(sqlite3_column_text16(stmt, 6));
                if (cachePath) chunk.cache_path = cachePath;

                m_chunks.push_back(std::move(chunk));
            }
        }
        sqlite3_finalize(stmt);
    }

    void ProjectManager::LoadInputImages()
    {
        m_inputImages.clear();
        if (!m_db) return;

        sqlite3_stmt* stmt = nullptr;
        const char* sql = "SELECT id, file_path, h00, h01, h02, h10, h11, h12, h20, h21, h22, gain FROM input_images ORDER BY id;";
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) == SQLITE_OK)
        {
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                InputImageModel img;
                img.id = sqlite3_column_int(stmt, 0);

                const wchar_t* path = reinterpret_cast<const wchar_t*>(sqlite3_column_text16(stmt, 1));
                if (path) img.file_path = path;

                img.homography.h[0] = static_cast<float>(sqlite3_column_double(stmt, 2));
                img.homography.h[1] = static_cast<float>(sqlite3_column_double(stmt, 3));
                img.homography.h[2] = static_cast<float>(sqlite3_column_double(stmt, 4));
                img.homography.h[3] = static_cast<float>(sqlite3_column_double(stmt, 5));
                img.homography.h[4] = static_cast<float>(sqlite3_column_double(stmt, 6));
                img.homography.h[5] = static_cast<float>(sqlite3_column_double(stmt, 7));
                img.homography.h[6] = static_cast<float>(sqlite3_column_double(stmt, 8));
                img.homography.h[7] = static_cast<float>(sqlite3_column_double(stmt, 9));
                img.homography.h[8] = static_cast<float>(sqlite3_column_double(stmt, 10));

                img.gain = static_cast<float>(sqlite3_column_double(stmt, 11));

                m_inputImages.push_back(std::move(img));
            }
        }
        sqlite3_finalize(stmt);
    }
}
