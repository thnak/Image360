#include "pch.h"
#include "HeaderFiles/TaskScheduler.h"
#include <future>
#include <list>
#include <vector>

namespace WindowsApp::Core
{
    TaskScheduler::TaskScheduler(ProjectManager& projectManager)
        : m_projectManager(projectManager)
    {
    }

    void TaskScheduler::RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor)
    {
        m_executors[stage] = std::move(executor);
    }

    bool TaskScheduler::RunStage(PipelineStage stage, CancellationToken token,
                                  std::function<void(const Task&, float stageProgress)> onTaskProgress)
    {
        // Every call, not just the first one for a project - resuming
        // mid-stage after a crash must also reclaim stale RUNNING rows.
        m_projectManager.ReclaimStaleRunningTasks(stage);

        auto executorIt = m_executors.find(stage);
        if (executorIt == m_executors.end()) return false;
        std::shared_ptr<ITaskExecutor> executor = executorIt->second;

        std::vector<Task> allTasks = m_projectManager.GetTasksForStage(stage);
        const size_t totalCount = allTasks.size();

        std::vector<Task> pending;
        size_t completedCount = 0;
        for (auto& task : allTasks)
        {
            if (task.status == TaskStatus::COMPLETED) ++completedCount;
            else pending.push_back(std::move(task));
        }

        // std::list so in-flight Task objects have a stable address even
        // as this container grows - the async lambda below holds a raw
        // pointer into it for the lifetime of the future.
        struct InFlight
        {
            Task task;
            std::future<bool> future;
        };
        std::list<InFlight> inFlight;

        auto dispatch = [&](Task task)
        {
            task.status = TaskStatus::RUNNING;
            m_projectManager.UpdateTaskStatus(task.taskId, TaskStatus::RUNNING);

            inFlight.push_back(InFlight{ std::move(task), std::future<bool>{} });
            InFlight& slot = inFlight.back();
            Task* taskPtr = &slot.task;

            slot.future = std::async(std::launch::async,
                [executor, taskPtr, token]()
                {
                    return executor->Execute(*taskPtr, token);
                });
        };

        bool anyFailedPermanently = false;

        auto settleFront = [&]()
        {
            InFlight& item = inFlight.front();
            bool ok = false;
            try { ok = item.future.get(); }
            catch (...) { ok = false; }

            if (ok)
            {
                if (item.task.outputBlobId.has_value())
                    m_projectManager.CommitTaskOutput(item.task.taskId, item.task.outputBlobId.value());
                else
                    m_projectManager.UpdateTaskStatus(item.task.taskId, TaskStatus::COMPLETED);
                ++completedCount;
            }
            else
            {
                int newAttemptCount = item.task.attemptCount + 1;
                if (newAttemptCount < kMaxAttempts)
                {
                    m_projectManager.UpdateTaskStatus(item.task.taskId, TaskStatus::PENDING);
                    Task retryTask = item.task;
                    retryTask.attemptCount = newAttemptCount;
                    retryTask.status = TaskStatus::PENDING;
                    pending.push_back(std::move(retryTask));
                }
                else
                {
                    m_projectManager.UpdateTaskStatus(item.task.taskId, TaskStatus::FAILED);
                    anyFailedPermanently = true;
                }
            }

            if (onTaskProgress)
            {
                float progress = totalCount > 0
                    ? static_cast<float>(completedCount) / static_cast<float>(totalCount)
                    : 1.0f;
                onTaskProgress(item.task, progress);
            }

            inFlight.pop_front();
        };

        size_t nextIndex = 0;
        bool cancelled = false;

        while (nextIndex < pending.size() || !inFlight.empty())
        {
            while (inFlight.size() < kMaxInFlight && nextIndex < pending.size())
            {
                if (token.stop_requested())
                {
                    cancelled = true;
                    break;
                }
                dispatch(pending[nextIndex]);
                ++nextIndex;
            }

            if (cancelled) break;

            if (!inFlight.empty())
            {
                settleFront();
            }
            else
            {
                break;
            }
        }

        // Cancellation only ever stops *new* dispatches - in-flight work
        // always finishes and commits, so drain it before returning.
        while (!inFlight.empty())
        {
            settleFront();
        }

        return !cancelled && !anyFailedPermanently;
    }
}
