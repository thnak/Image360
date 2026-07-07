#include "CppUnitTest.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/ProjectManager.h"
#include "StubTaskExecutor.h"
#include <Windows.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

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

        std::vector<WindowsApp::Core::Task> MakeAlignTasks(int count)
        {
            std::vector<WindowsApp::Core::Task> tasks;
            for (int i = 0; i < count; ++i)
            {
                WindowsApp::Core::Task t;
                t.stage = WindowsApp::Core::PipelineStage::STAGE1_ALIGN;
                t.unitKind = "image";
                t.unitKey = "img_" + std::to_string(i);
                tasks.push_back(t);
            }
            return tasks;
        }
    }

    // Covers docs/superpowers/plans/2026-07-07-task-scheduler-core.md Task 5:
    // proves PipelineDriver/TaskScheduler's resume and cancellation
    // behavior against StubTaskExecutor, with no GPU dependency.
    TEST_CLASS(PipelineDriverTests)
    {
    public:
        TEST_METHOD(FreshRunCompletesAllSeededTasks)
        {
            using namespace WindowsApp::Core;

            std::wstring path = MakeTempProjectPath(L"pd_freshrun");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Assert::IsTrue(pm.CreateTasksIfAbsent(MakeAlignTasks(2)));

            Task optimizeTask;
            optimizeTask.stage = PipelineStage::STAGE2_OPTIMIZE;
            optimizeTask.unitKind = "pair";
            optimizeTask.unitKey = "img_0:img_1";
            Assert::IsTrue(pm.CreateTasksIfAbsent({ optimizeTask }));

            auto alignExecutor = std::make_shared<StubTaskExecutor>();
            auto optimizeExecutor = std::make_shared<StubTaskExecutor>();

            PipelineDriver driver;
            driver.Initialize(nullptr, nullptr);
            driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN, alignExecutor);
            driver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE, optimizeExecutor);

            std::stop_source source;
            Assert::IsTrue(driver.Run(pm, source.get_token()));

            for (const auto& task : pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN))
                Assert::IsTrue(task.status == TaskStatus::COMPLETED);
            for (const auto& task : pm.GetTasksForStage(PipelineStage::STAGE2_OPTIMIZE))
                Assert::IsTrue(task.status == TaskStatus::COMPLETED);

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }

        TEST_METHOD(ResumeSkipsAlreadyCompletedTasks)
        {
            using namespace WindowsApp::Core;

            std::wstring path = MakeTempProjectPath(L"pd_resume");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Assert::IsTrue(pm.CreateTasksIfAbsent(MakeAlignTasks(4)));

            auto tasks = pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN);
            Assert::AreEqual(size_t(4), tasks.size());
            // Simulate a prior run that already finished half of this stage.
            Assert::IsTrue(pm.CommitTaskOutput(tasks[0].taskId, 1));
            Assert::IsTrue(pm.CommitTaskOutput(tasks[1].taskId, 2));

            auto executor = std::make_shared<StubTaskExecutor>();

            PipelineDriver driver;
            driver.Initialize(nullptr, nullptr);
            driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN, executor);

            std::stop_source source;
            Assert::IsTrue(driver.Run(pm, source.get_token()));

            Assert::AreEqual(2, executor->CompletionCount());

            for (const auto& task : pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN))
                Assert::IsTrue(task.status == TaskStatus::COMPLETED);

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }

        TEST_METHOD(CrashSimulationReclaimsAndReExecutesRunningTask)
        {
            using namespace WindowsApp::Core;

            std::wstring path = MakeTempProjectPath(L"pd_crash");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            Assert::IsTrue(pm.CreateTasksIfAbsent(MakeAlignTasks(2)));

            auto tasks = pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN);
            Assert::AreEqual(size_t(2), tasks.size());
            // Simulate a prior process that started img_0 and died before committing.
            Assert::IsTrue(pm.UpdateTaskStatus(tasks[0].taskId, TaskStatus::RUNNING));

            auto executor = std::make_shared<StubTaskExecutor>();

            PipelineDriver driver;
            driver.Initialize(nullptr, nullptr);
            driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN, executor);

            std::stop_source source;
            Assert::IsTrue(driver.Run(pm, source.get_token()));

            Assert::AreEqual(2, executor->CompletionCount());
            for (const auto& task : pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN))
                Assert::IsTrue(task.status == TaskStatus::COMPLETED);

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }

        TEST_METHOD(CancellationStopsNewDispatchButFinishesInFlightTasks)
        {
            using namespace WindowsApp::Core;

            std::wstring path = MakeTempProjectPath(L"pd_cancel");
            ProjectManager pm;
            Assert::IsTrue(pm.CreateProject(path, 8192, 8192));

            // More than kMaxInFlight (4) so some tasks are guaranteed to
            // never be dispatched before the stop request lands.
            Assert::IsTrue(pm.CreateTasksIfAbsent(MakeAlignTasks(12)));

            auto executor = std::make_shared<StubTaskExecutor>(std::chrono::milliseconds(150));

            PipelineDriver driver;
            driver.Initialize(nullptr, nullptr);
            driver.RegisterExecutor(PipelineStage::STAGE1_ALIGN, executor);

            std::stop_source source;
            std::atomic<bool> runResult{ true };

            std::thread worker([&]()
            {
                runResult.store(driver.Run(pm, source.get_token()));
            });

            // Long enough for the first (kMaxInFlight) batch to have
            // dispatched (dispatch itself doesn't block), short enough
            // that none of the 150ms stub tasks have settled yet.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            source.request_stop();
            worker.join();

            Assert::IsFalse(runResult.load());

            int completedCount = 0;
            int pendingCount = 0;
            int otherCount = 0;
            for (const auto& task : pm.GetTasksForStage(PipelineStage::STAGE1_ALIGN))
            {
                if (task.status == TaskStatus::COMPLETED) ++completedCount;
                else if (task.status == TaskStatus::PENDING) ++pendingCount;
                else ++otherCount;
            }

            // Never left RUNNING or half-done.
            Assert::AreEqual(0, otherCount);
            // The batch that was already in-flight when the stop request
            // landed always finishes and commits.
            Assert::IsTrue(completedCount >= 1);
            // Every task ends up either COMPLETED (was in-flight) or still
            // PENDING (never dispatched) - none lost.
            Assert::AreEqual(12, completedCount + pendingCount);

            pm.CloseProject();
            DeleteFileW(path.c_str());
        }
    };
}
