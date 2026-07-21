#include "pch.h"
#include "HeaderFiles/ProjectManager.h"
#include "HeaderFiles/OverlapCulling.h"
#include "HeaderFiles/TextEncoding.h"
#include "sqlite3/sqlite3.h"
#include <stdexcept>

namespace WindowsApp::Core
{
    namespace
    {
        const char* ToString(PipelineStage stage)
        {
            switch (stage)
            {
            case PipelineStage::IDLE:            return "IDLE";
            case PipelineStage::STAGE0_INGEST:   return "STAGE0_INGEST";
            case PipelineStage::STAGE1_ALIGN:    return "STAGE1_ALIGN";
            case PipelineStage::STAGE2_OPTIMIZE: return "STAGE2_OPTIMIZE";
            case PipelineStage::STAGE3_RENDER:   return "STAGE3_RENDER";
            case PipelineStage::BURST_ALIGN:     return "BURST_ALIGN";
            case PipelineStage::BURST_MERGE:     return "BURST_MERGE";
            case PipelineStage::BURST_FINISH:    return "BURST_FINISH";
            case PipelineStage::COMPLETED:       return "COMPLETED";
            case PipelineStage::CANCELLED:       return "CANCELLED";
            case PipelineStage::FAILED:          return "FAILED";
            }
            return "IDLE";
        }

        PipelineStage ParsePipelineStage(const std::string& s)
        {
            if (s == "STAGE0_INGEST")   return PipelineStage::STAGE0_INGEST;
            if (s == "STAGE1_ALIGN")    return PipelineStage::STAGE1_ALIGN;
            if (s == "STAGE2_OPTIMIZE") return PipelineStage::STAGE2_OPTIMIZE;
            if (s == "STAGE3_RENDER")   return PipelineStage::STAGE3_RENDER;
            if (s == "BURST_ALIGN")     return PipelineStage::BURST_ALIGN;
            if (s == "BURST_MERGE")     return PipelineStage::BURST_MERGE;
            if (s == "BURST_FINISH")    return PipelineStage::BURST_FINISH;
            if (s == "COMPLETED")       return PipelineStage::COMPLETED;
            if (s == "CANCELLED")       return PipelineStage::CANCELLED;
            if (s == "FAILED")          return PipelineStage::FAILED;
            return PipelineStage::IDLE;
        }

        const char* ToString(ProjectType type)
        {
            switch (type)
            {
            case ProjectType::PANORAMA: return "PANORAMA";
            case ProjectType::BURST:    return "BURST";
            }
            return "PANORAMA";
        }

        ProjectType ParseProjectType(const std::string& s)
        {
            if (s == "BURST") return ProjectType::BURST;
            return ProjectType::PANORAMA;
        }

        const char* ToString(BurstMode mode)
        {
            switch (mode)
            {
            case BurstMode::NONE:        return "NONE";
            case BurstMode::MFNR:        return "MFNR";
            case BurstMode::HDR_PLUS:    return "HDR_PLUS";
            case BurstMode::NIGHT_SIGHT: return "NIGHT_SIGHT";
            case BurstMode::SUPER_RES:   return "SUPER_RES";
            }
            return "NONE";
        }

        BurstMode ParseBurstMode(const std::string& s)
        {
            if (s == "MFNR")        return BurstMode::MFNR;
            if (s == "HDR_PLUS")    return BurstMode::HDR_PLUS;
            if (s == "NIGHT_SIGHT") return BurstMode::NIGHT_SIGHT;
            if (s == "SUPER_RES")   return BurstMode::SUPER_RES;
            return BurstMode::NONE;
        }

        const char* ToString(TaskStatus status)
        {
            switch (status)
            {
            case TaskStatus::PENDING:   return "PENDING";
            case TaskStatus::RUNNING:   return "RUNNING";
            case TaskStatus::COMPLETED: return "COMPLETED";
            case TaskStatus::FAILED:    return "FAILED";
            case TaskStatus::CANCELLED: return "CANCELLED";
            }
            return "PENDING";
        }

        TaskStatus ParseTaskStatus(const std::string& s)
        {
            if (s == "RUNNING")   return TaskStatus::RUNNING;
            if (s == "COMPLETED") return TaskStatus::COMPLETED;
            if (s == "FAILED")    return TaskStatus::FAILED;
            if (s == "CANCELLED") return TaskStatus::CANCELLED;
            return TaskStatus::PENDING;
        }

        const char* ToString(CfaType cfaType)
        {
            switch (cfaType)
            {
            case CfaType::BAYER:        return "BAYER";
            case CfaType::X_TRANS:      return "X_TRANS";
            case CfaType::FOVEON:       return "FOVEON";
            case CfaType::STANDARD_RGB: return "STANDARD_RGB";
            case CfaType::UNKNOWN:      return "UNKNOWN";
            }
            return "UNKNOWN";
        }

        CfaType ParseCfaType(const std::string& s)
        {
            if (s == "BAYER")        return CfaType::BAYER;
            if (s == "X_TRANS")      return CfaType::X_TRANS;
            if (s == "FOVEON")       return CfaType::FOVEON;
            if (s == "STANDARD_RGB") return CfaType::STANDARD_RGB;
            return CfaType::UNKNOWN;
        }
    }

    ProjectManager::ProjectManager() = default;

    ProjectManager::~ProjectManager()
    {
        CloseProject();
    }

    int RecommendedChunkSize(uint64_t totalVramBytes)
    {
        constexpr uint64_t kGB = 1024ull * 1024 * 1024;
        if (totalVramBytes >= 16ull * kGB) return 4096;
        if (totalVramBytes >= 8ull * kGB) return 2048;
        return 1024;
    }

    int RecommendedChunkSizeForCpu(uint64_t totalRamBytes)
    {
        constexpr uint64_t kGB = 1024ull * 1024 * 1024;
        if (totalRamBytes >= 32ull * kGB) return 4096;
        if (totalRamBytes >= 16ull * kGB) return 2048;
        return 1024;
    }

    bool ProjectManager::OpenAndCreateSchema(const std::wstring& dbPath)
    {
        CloseProject();

        // Convert wide path to UTF-8 for SQLite
        std::string utf8Path = WideToUtf8(dbPath);

        int rc = sqlite3_open(utf8Path.c_str(), &m_db);
        if (rc != SQLITE_OK)
        {
            m_db = nullptr;
            return false;
        }

        // Enable WAL mode for better concurrent read/write performance
        ExecuteNonQuery("PRAGMA journal_mode=WAL;");
        ExecuteNonQuery("PRAGMA synchronous=NORMAL;");

        // Create schema - every table both ProjectTypes need. chunks/
        // chunk_contributors stay unused/empty for a burst project, same as
        // tasks/blob_directory stay unused/empty for a never-run panorama
        // project - simpler than a schema that varies by ProjectType.
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
                gain REAL DEFAULT 1.0,
                cfa_type TEXT DEFAULT 'BAYER'
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

        return true;
    }

    bool ProjectManager::CreateProject(const std::wstring& dbPath, int totalWidth, int totalHeight, int chunkSize)
    {
        if (!OpenAndCreateSchema(dbPath))
            return false;

        // Store metadata
        std::string sql = "INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('total_width', '"
            + std::to_string(totalWidth) + "');";
        ExecuteNonQuery(sql);

        sql = "INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('total_height', '"
            + std::to_string(totalHeight) + "');";
        ExecuteNonQuery(sql);

        ExecuteNonQuery("INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('project_type', 'PANORAMA');");
        ExecuteNonQuery("INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('burst_mode', 'NONE');");
        m_projectType = ProjectType::PANORAMA;
        m_burstMode = BurstMode::NONE;

        m_projectPath = dbPath;
        m_totalWidth = totalWidth;
        m_totalHeight = totalHeight;
        m_chunks.clear();
        m_inputImages.clear();

        // Generate chunk grid (chunkSize x chunkSize tiles - VRAM-budget-
        // derived by the caller via RecommendedChunkSize, defaulted to
        // 4096 so every existing call site keeps compiling unchanged).
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

    bool ProjectManager::CreateBurstProject(const std::wstring& dbPath, BurstMode mode)
    {
        if (!OpenAndCreateSchema(dbPath))
            return false;

        ExecuteNonQuery("INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('project_type', 'BURST');");
        ExecuteNonQuery(std::string("INSERT OR REPLACE INTO project_metadata (key, value) VALUES ('burst_mode', '")
            + ToString(mode) + "');");

        m_projectType = ProjectType::BURST;
        m_burstMode = mode;
        m_projectPath = dbPath;
        // No chunk grid: a burst project's output is one merged frame, not
        // a tiled canvas. total_width/total_height stay 0 (and unwritten to
        // project_metadata) until a real burst executor sets them from the
        // first ingested frame - out of scope for this foundation slice.
        m_totalWidth = 0;
        m_totalHeight = 0;
        m_chunks.clear();
        m_inputImages.clear();

        return true;
    }

    bool ProjectManager::LoadProject(const std::wstring& dbPath)
    {
        CloseProject();

        std::string utf8Path = WideToUtf8(dbPath);

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

        // ALTER TABLE ADD COLUMN fails if the column already exists (a
        // project created after cfa_type was added to the schema above) -
        // that failure just means "already migrated," not an error, so
        // its return value is intentionally ignored here.
        ExecuteNonQuery("ALTER TABLE input_images ADD COLUMN cfa_type TEXT DEFAULT 'BAYER';");

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
        m_projectType = ProjectType::PANORAMA;
        m_burstMode = BurstMode::NONE;
        m_chunks.clear();
        m_inputImages.clear();
    }

    bool ProjectManager::AddInputImage(const std::wstring& filePath, const Homography& h, CfaType cfaType)
    {
        if (!m_db) return false;

        std::string utf8Path = WideToUtf8(filePath);

        char sql[1024];
        snprintf(sql, sizeof(sql),
            "INSERT INTO input_images (file_path, h00, h01, h02, h10, h11, h12, h20, h21, h22, cfa_type) "
            "VALUES ('%s', %f, %f, %f, %f, %f, %f, %f, %f, %f, '%s');",
            utf8Path.c_str(),
            h.h[0], h.h[1], h.h[2],
            h.h[3], h.h[4], h.h[5],
            h.h[6], h.h[7], h.h[8],
            ToString(cfaType));

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

    bool ProjectManager::UpdateHomography(int imageId, const Homography& h)
    {
        if (!m_db) return false;

        const char* sql =
            "UPDATE input_images SET h00=?, h01=?, h02=?, h10=?, h11=?, h12=?, h20=?, h21=?, h22=? WHERE id=?;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        for (int i = 0; i < 9; ++i)
        {
            sqlite3_bind_double(stmt, i + 1, static_cast<double>(h.h[i]));
        }
        sqlite3_bind_int(stmt, 10, imageId);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);

        if (ok)
        {
            for (auto& img : m_inputImages)
            {
                if (img.id == imageId)
                {
                    img.homography = h;
                    break;
                }
            }
        }

        return ok;
    }

    bool ProjectManager::SeedIngestTasks()
    {
        if (!m_db) return false;

        std::vector<Task> seeds;
        seeds.reserve(m_inputImages.size());
        for (const auto& image : m_inputImages)
        {
            Task task;
            task.stage = PipelineStage::STAGE0_INGEST;
            task.unitKind = "image";
            task.unitKey = std::to_string(image.id);
            seeds.push_back(std::move(task));
        }

        return CreateTasksIfAbsent(seeds);
    }

    bool ProjectManager::SeedBurstAlignTasks()
    {
        if (!m_db) return false;

        std::vector<Task> seeds;
        seeds.reserve(m_inputImages.size());
        for (const auto& image : m_inputImages)
        {
            Task task;
            task.stage = PipelineStage::BURST_ALIGN;
            task.unitKind = "frame";
            task.unitKey = std::to_string(image.id);
            seeds.push_back(std::move(task));
        }

        return CreateTasksIfAbsent(seeds);
    }

    bool ProjectManager::SeedBurstMergeTasks()
    {
        if (!m_db) return false;

        // Both safe to seed upfront alongside SeedBurstAlignTasks - unlike
        // panorama's STAGE3_RENDER, neither depends on BURST_ALIGN's
        // *results* (only that it has run first, which TaskScheduler's
        // stage ordering already guarantees) - see
        // docs/superpowers/plans/2026-07-21-mfnr-block-match-merge.md's
        // Architecture note.
        Task mergeTask;
        mergeTask.stage = PipelineStage::BURST_MERGE;
        mergeTask.unitKind = "output";
        mergeTask.unitKey = "merged";

        Task finishTask;
        finishTask.stage = PipelineStage::BURST_FINISH;
        finishTask.unitKind = "output";
        finishTask.unitKey = "final";

        return CreateTasksIfAbsent({ mergeTask, finishTask });
    }

    bool ProjectManager::SeedAlignTasks()
    {
        if (!m_db) return false;

        std::vector<Task> seeds;
        seeds.reserve(m_inputImages.size());
        for (const auto& image : m_inputImages)
        {
            Task task;
            task.stage = PipelineStage::STAGE1_ALIGN;
            task.unitKind = "image";
            task.unitKey = std::to_string(image.id);
            seeds.push_back(std::move(task));
        }

        for (size_t i = 0; i < m_inputImages.size(); ++i)
        {
            for (size_t j = i + 1; j < m_inputImages.size(); ++j)
            {
                Task task;
                task.stage = PipelineStage::STAGE1_ALIGN;
                task.unitKind = "pair";
                task.unitKey = std::to_string(m_inputImages[i].id) + ":" + std::to_string(m_inputImages[j].id);
                seeds.push_back(std::move(task));
            }
        }

        return CreateTasksIfAbsent(seeds);
    }

    bool ProjectManager::SeedOptimizeTasks()
    {
        if (!m_db) return false;

        std::vector<Task> seeds;
        seeds.reserve(m_inputImages.size() * 2 + 1);
        for (const auto& image : m_inputImages)
        {
            Task gainTask;
            gainTask.stage = PipelineStage::STAGE2_OPTIMIZE;
            gainTask.unitKind = "gain";
            gainTask.unitKey = std::to_string(image.id);
            seeds.push_back(std::move(gainTask));

            Task colorTask;
            colorTask.stage = PipelineStage::STAGE2_OPTIMIZE;
            colorTask.unitKind = "color";
            colorTask.unitKey = std::to_string(image.id);
            seeds.push_back(std::move(colorTask));
        }

        Task baTask;
        baTask.stage = PipelineStage::STAGE2_OPTIMIZE;
        baTask.unitKind = "ba_checkpoint";
        baTask.unitKey = "global";
        seeds.push_back(std::move(baTask));

        return CreateTasksIfAbsent(seeds);
    }

    bool ProjectManager::SeedRenderTasks()
    {
        if (!m_db) return false;

        std::vector<Task> seeds;
        for (const auto& chunk : m_chunks)
        {
            std::vector<int> contributors = FindOverlappingImages(chunk, m_inputImages);
            SetChunkContributors(chunk.id, contributors);

            if (!contributors.empty())
            {
                Task task;
                task.stage = PipelineStage::STAGE3_RENDER;
                task.unitKind = "chunk";
                task.unitKey = chunk.id;
                seeds.push_back(std::move(task));
            }
        }

        return CreateTasksIfAbsent(seeds);
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

        std::string utf8Cache = WideToUtf8(cachePath);

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

    bool ProjectManager::CreateTasksIfAbsent(const std::vector<Task>& tasks)
    {
        if (!m_db) return false;

        const char* sql =
            "INSERT OR IGNORE INTO tasks (stage, unit_kind, unit_key, status, attempt_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, strftime('%s','now'));";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        bool ok = true;
        for (const auto& task : tasks)
        {
            sqlite3_bind_text(stmt, 1, ToString(task.stage), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 2, task.unitKind.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 3, task.unitKey.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(stmt, 4, ToString(task.status), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(stmt, 5, task.attemptCount);

            if (sqlite3_step(stmt) != SQLITE_DONE) ok = false;
            sqlite3_reset(stmt);
        }

        sqlite3_finalize(stmt);
        return ok;
    }

    std::vector<Task> ProjectManager::GetTasksForStage(PipelineStage stage) const
    {
        std::vector<Task> results;
        if (!m_db) return results;

        const char* sql =
            "SELECT task_id, stage, unit_kind, unit_key, status, attempt_count, output_blob_id, checkpoint_json "
            "FROM tasks WHERE stage = ? ORDER BY task_id;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

        sqlite3_bind_text(stmt, 1, ToString(stage), -1, SQLITE_TRANSIENT);

        while (sqlite3_step(stmt) == SQLITE_ROW)
        {
            Task task;
            task.taskId = sqlite3_column_int64(stmt, 0);
            task.stage = ParsePipelineStage(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1)));
            task.unitKind = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            task.unitKey = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
            task.status = ParseTaskStatus(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4)));
            task.attemptCount = sqlite3_column_int(stmt, 5);

            if (sqlite3_column_type(stmt, 6) != SQLITE_NULL)
                task.outputBlobId = sqlite3_column_int64(stmt, 6);

            if (const char* checkpoint = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7)))
                task.checkpointJson = checkpoint;

            results.push_back(std::move(task));
        }

        sqlite3_finalize(stmt);
        return results;
    }

    bool ProjectManager::UpdateTaskStatus(int64_t taskId, TaskStatus status)
    {
        if (!m_db) return false;

        const char* sql = "UPDATE tasks SET status = ?, updated_at = strftime('%s','now') WHERE task_id = ?;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, ToString(status), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, taskId);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool ProjectManager::CommitTaskOutput(int64_t taskId, int64_t outputBlobId)
    {
        if (!m_db) return false;

        const char* sql =
            "UPDATE tasks SET status = 'COMPLETED', output_blob_id = ?, updated_at = strftime('%s','now') "
            "WHERE task_id = ?;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_int64(stmt, 1, outputBlobId);
        sqlite3_bind_int64(stmt, 2, taskId);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    bool ProjectManager::UpdateTaskCheckpoint(int64_t taskId, const std::string& checkpointJson)
    {
        if (!m_db) return false;

        const char* sql = "UPDATE tasks SET checkpoint_json = ?, updated_at = strftime('%s','now') WHERE task_id = ?;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return false;

        sqlite3_bind_text(stmt, 1, checkpointJson.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, taskId);

        bool ok = sqlite3_step(stmt) == SQLITE_DONE;
        sqlite3_finalize(stmt);
        return ok;
    }

    int ProjectManager::ReclaimStaleRunningTasks(PipelineStage stage)
    {
        if (!m_db) return 0;

        const char* sql = "UPDATE tasks SET status = 'PENDING' WHERE stage = ? AND status = 'RUNNING';";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;

        sqlite3_bind_text(stmt, 1, ToString(stage), -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);

        return sqlite3_changes(m_db);
    }

    bool ProjectManager::SetChunkContributors(const std::string& chunkId, const std::vector<int>& imageIds)
    {
        if (!m_db) return false;

        sqlite3_stmt* del = nullptr;
        if (sqlite3_prepare_v2(m_db, "DELETE FROM chunk_contributors WHERE chunk_id = ?;", -1, &del, nullptr) != SQLITE_OK)
            return false;
        sqlite3_bind_text(del, 1, chunkId.c_str(), -1, SQLITE_TRANSIENT);
        bool ok = sqlite3_step(del) == SQLITE_DONE;
        sqlite3_finalize(del);
        if (!ok) return false;

        sqlite3_stmt* ins = nullptr;
        if (sqlite3_prepare_v2(m_db, "INSERT INTO chunk_contributors (chunk_id, image_id) VALUES (?, ?);", -1, &ins, nullptr) != SQLITE_OK)
            return false;

        for (int imageId : imageIds)
        {
            sqlite3_bind_text(ins, 1, chunkId.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_int(ins, 2, imageId);

            if (sqlite3_step(ins) != SQLITE_DONE) ok = false;
            sqlite3_reset(ins);
        }

        sqlite3_finalize(ins);
        return ok;
    }

    std::vector<int> ProjectManager::GetChunkContributors(const std::string& chunkId) const
    {
        std::vector<int> results;
        if (!m_db) return results;

        const char* sql = "SELECT image_id FROM chunk_contributors WHERE chunk_id = ? ORDER BY image_id;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return results;

        sqlite3_bind_text(stmt, 1, chunkId.c_str(), -1, SQLITE_TRANSIENT);

        while (sqlite3_step(stmt) == SQLITE_ROW)
            results.push_back(sqlite3_column_int(stmt, 0));

        sqlite3_finalize(stmt);
        return results;
    }

    int64_t ProjectManager::AddBlobDirectoryEntry(const BlobDirectoryEntry& entry)
    {
        if (!m_db) return 0;

        const char* sql =
            "INSERT INTO blob_directory (shard_file, offset, length, compressed_length, format_tag) "
            "VALUES (?, ?, ?, ?, ?);";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return 0;

        std::string utf8Shard = WideToUtf8(entry.shardFile);

        sqlite3_bind_text(stmt, 1, utf8Shard.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(stmt, 2, entry.offset);
        sqlite3_bind_int64(stmt, 3, entry.length);
        if (entry.compressedLength.has_value())
            sqlite3_bind_int64(stmt, 4, entry.compressedLength.value());
        else
            sqlite3_bind_null(stmt, 4);
        sqlite3_bind_text(stmt, 5, entry.formatTag.c_str(), -1, SQLITE_TRANSIENT);

        int64_t newId = 0;
        if (sqlite3_step(stmt) == SQLITE_DONE)
            newId = sqlite3_last_insert_rowid(m_db);

        sqlite3_finalize(stmt);
        return newId;
    }

    std::optional<BlobDirectoryEntry> ProjectManager::GetBlobDirectoryEntry(int64_t blobId) const
    {
        if (!m_db) return std::nullopt;

        const char* sql =
            "SELECT blob_id, shard_file, offset, length, compressed_length, format_tag "
            "FROM blob_directory WHERE blob_id = ?;";

        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) return std::nullopt;

        sqlite3_bind_int64(stmt, 1, blobId);

        std::optional<BlobDirectoryEntry> result;
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            BlobDirectoryEntry entry;
            entry.blobId = sqlite3_column_int64(stmt, 0);

            const char* shardFileUtf8 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            if (shardFileUtf8)
            {
                entry.shardFile = Utf8ToWide(shardFileUtf8);
            }

            entry.offset = sqlite3_column_int64(stmt, 2);
            entry.length = sqlite3_column_int64(stmt, 3);

            if (sqlite3_column_type(stmt, 4) != SQLITE_NULL)
                entry.compressedLength = sqlite3_column_int64(stmt, 4);

            if (const char* formatTag = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5)))
                entry.formatTag = formatTag;

            result = std::move(entry);
        }

        sqlite3_finalize(stmt);
        return result;
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
                    // Missing key (project created before ProjectType/
                    // BurstMode existed) leaves the constructor defaults
                    // (PANORAMA/NONE) untouched - no migration needed.
                    else if (k == "project_type") m_projectType = ParseProjectType(value);
                    else if (k == "burst_mode") m_burstMode = ParseBurstMode(value);
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

                const char* cachePathUtf8 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
                if (cachePathUtf8)
                {
                    chunk.cache_path = Utf8ToWide(cachePathUtf8);
                }

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
        const char* sql = "SELECT id, file_path, h00, h01, h02, h10, h11, h12, h20, h21, h22, gain, cfa_type FROM input_images ORDER BY id;";
        if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) == SQLITE_OK)
        {
            while (sqlite3_step(stmt) == SQLITE_ROW)
            {
                InputImageModel img;
                img.id = sqlite3_column_int(stmt, 0);

                const char* pathUtf8 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                if (pathUtf8)
                {
                    img.file_path = Utf8ToWide(pathUtf8);
                }

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

                if (const char* cfaTypeUtf8 = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 12)))
                    img.cfaType = ParseCfaType(cfaTypeUtf8);

                m_inputImages.push_back(std::move(img));
            }
        }
        sqlite3_finalize(stmt);
    }
}
