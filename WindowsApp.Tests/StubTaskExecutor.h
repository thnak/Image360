#pragma once
#include "HeaderFiles/ITaskExecutor.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace WindowsApp::Tests
{
    // Configurable ITaskExecutor test double: simulates variable-duration
    // async work and honors cancellation only at its own entry point -
    // once dispatched, a unit of work always runs to completion, per
    // docs/ARCHITECTURE.md SS7.2's "in-flight work always finishes" rule.
    class StubTaskExecutor : public WindowsApp::Core::ITaskExecutor
    {
    public:
        explicit StubTaskExecutor(std::chrono::milliseconds duration = std::chrono::milliseconds(0))
            : m_duration(duration)
        {
        }

        // Makes Execute() fail the next `failCount` invocations for
        // `unitKey` (across retries), succeeding once the budget is spent -
        // lets a test exercise TaskScheduler's retry path deterministically.
        void SetFailFirstAttempts(const std::string& unitKey, int failCount)
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_failBudget[unitKey] = failCount;
        }

        int CompletionCount() const { return m_completionCount.load(); }

        bool Execute(WindowsApp::Core::Task& task, WindowsApp::Core::CancellationToken token) override
        {
            // Checked once at entry only - not acted on. A real GPU-graph
            // executor works the same way: cancellation decides whether a
            // task is *dispatched* at all, never aborts one already running.
            (void)token.stop_requested();

            if (m_duration.count() > 0)
                std::this_thread::sleep_for(m_duration);

            bool shouldFail = false;
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = m_failBudget.find(task.unitKey);
                if (it != m_failBudget.end() && it->second > 0)
                {
                    --it->second;
                    shouldFail = true;
                }
            }

            if (shouldFail) return false;

            m_completionCount.fetch_add(1);
            return true;
        }

    private:
        std::chrono::milliseconds m_duration;
        std::atomic<int> m_completionCount{ 0 };
        std::mutex m_mutex;
        std::unordered_map<std::string, int> m_failBudget;
    };
}
