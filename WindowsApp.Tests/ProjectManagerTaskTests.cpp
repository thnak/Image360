#include "CppUnitTest.h"
#include "ProjectManager.h"
#include <Windows.h>
#include <cstdint>
#include <string>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace WindowsApp::Tests
{
    namespace
    {
        std::wstring MakeTempProjectPath(const wchar_t* suffix)
        {
            wchar_t tempDir[MAX_PATH];
            GetTempPathW(MAX_PATH, tempDir);
            std::wstring path = std::wstring(tempDir) + L"Image360Test_" + suffix + L".vfp";
            DeleteFileW(path.c_str()); // clean up any leftover from a prior failed run
            return path;
        }
    }

    // Covers docs/superpowers/plans/2026-07-07-vfp-project-schema.md Task 5:
    // schema smoke test, Task CRUD round-trip, UNIQUE re-seed, and the
    // crash-recovery rule from docs/ARCHITECTURE.md SS7.2.
    TEST_CLASS(ProjectManagerTaskTests)
    {
    public:
        TEST_METHOD(FreshProjectHasEmptyTaskTables)
        {
            using namespace WindowsApp::Core;
            std::wstring path = MakeTempProjectPath(L"schema");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Assert::IsTrue(pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN).empty());
            Assert::IsTrue(pm.GetChunkContributors("C_0_0").empty());
            Assert::IsFalse(pm.GetBlobDirectoryEntry(1).has_value());

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }

        TEST_METHOD(TaskCrudRoundTrip)
        {
            using namespace WindowsApp::Core;
            std::wstring path = MakeTempProjectPath(L"crud");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Task t;
            t.stage = PipelineStage::STAGE1_ALIGN;
            t.unitKind = "image";
            t.unitKey = "img_1";
            Assert::IsTrue(pm.CreateTasksIfAbsent({ t }));

            auto tasks = pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN);
            Assert::AreEqual(size_t(1), tasks.size());
            Assert::IsTrue(tasks[0].status == TaskStatus::PENDING);

            BlobDirectoryEntry entry;
            entry.shardFile = L"shard0.vfpdata";
            entry.offset = 0;
            entry.length = 1024;
            entry.formatTag = "raw_rgb48";
            int64_t blobId = pm.AddBlobDirectoryEntry(entry);
            Assert::IsTrue(blobId > 0);

            Assert::IsTrue(pm.CommitTaskOutput(tasks[0].taskId, blobId));

            auto reQueried = pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN);
            Assert::AreEqual(size_t(1), reQueried.size());
            Assert::IsTrue(reQueried[0].status == TaskStatus::COMPLETED);
            Assert::IsTrue(reQueried[0].outputBlobId.has_value());
            Assert::AreEqual(blobId, reQueried[0].outputBlobId.value());

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }

        TEST_METHOD(CreateTasksIfAbsentDoesNotDuplicateOnReseed)
        {
            using namespace WindowsApp::Core;
            std::wstring path = MakeTempProjectPath(L"reseed");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Task t;
            t.stage = PipelineStage::STAGE2_OPTIMIZE;
            t.unitKind = "pair";
            t.unitKey = "img_2:img_9";

            Assert::IsTrue(pm.CreateTasksIfAbsent({ t }));
            Assert::IsTrue(pm.CreateTasksIfAbsent({ t }));

            auto tasks = pm.GetTasksForStage(PipelineStage::STAGE2_OPTIMIZE);
            Assert::AreEqual(size_t(1), tasks.size());

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }

        TEST_METHOD(ReclaimStaleRunningTasksResetsToPending)
        {
            using namespace WindowsApp::Core;
            std::wstring path = MakeTempProjectPath(L"reclaim");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Task t;
            t.stage = PipelineStage::STAGE3_RENDER;
            t.unitKind = "chunk";
            t.unitKey = "C_0_0";
            Assert::IsTrue(pm.CreateTasksIfAbsent({ t }));

            auto tasks = pm.GetTasksForStage(PipelineStage::STAGE3_RENDER);
            Assert::AreEqual(size_t(1), tasks.size());
            Assert::IsTrue(pm.UpdateTaskStatus(tasks[0].taskId, TaskStatus::RUNNING));

            int reclaimed = pm.ReclaimStaleRunningTasks(PipelineStage::STAGE3_RENDER);
            Assert::AreEqual(1, reclaimed);

            auto reQueried = pm.GetTasksForStage(PipelineStage::STAGE3_RENDER);
            Assert::AreEqual(size_t(1), reQueried.size());
            Assert::IsTrue(reQueried[0].status == TaskStatus::PENDING);

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }
    };
}
