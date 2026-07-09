#include "pch.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/TaskScheduler.h"
#include <algorithm>
#include <iterator>
#include <vector>

namespace WindowsApp::Core
{
    namespace
    {
        const wchar_t* StageName(PipelineStage stage)
        {
            switch (stage)
            {
            case PipelineStage::STAGE0_INGEST:   return L"Ingest";
            case PipelineStage::STAGE1_ALIGN:    return L"Align";
            case PipelineStage::STAGE2_OPTIMIZE: return L"Optimize";
            case PipelineStage::STAGE3_RENDER:   return L"Render";
            default:                             return L"?";
            }
        }

        // unitKind/unitKey are ASCII ids (e.g. "pair", "img_2:img_9") -
        // see Types.h's Task comment - so a byte-for-byte widen is safe.
        std::wstring Widen(const std::string& s)
        {
            return std::wstring(s.begin(), s.end());
        }
    }

    void PipelineDriver::Initialize(ProgressCallback onProgress, LogCallback onLog, size_t maxInFlight)
    {
        m_onProgress = std::move(onProgress);
        m_onLog = std::move(onLog);
        m_maxInFlight = maxInFlight;
    }

    void PipelineDriver::RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor)
    {
        m_executors[stage] = std::move(executor);
    }

    bool PipelineDriver::Run(ProjectManager& projectManager, CancellationToken token)
    {
        static constexpr PipelineStage kStageOrder[] = {
            PipelineStage::STAGE0_INGEST,
            PipelineStage::STAGE1_ALIGN,
            PipelineStage::STAGE2_OPTIMIZE,
            PipelineStage::STAGE3_RENDER,
        };
        static constexpr size_t kStageCount = std::size(kStageOrder);

        TaskScheduler scheduler(projectManager, m_maxInFlight);
        for (const auto& [stage, executor] : m_executors)
            scheduler.RegisterExecutor(stage, executor);

        for (size_t i = 0; i < kStageCount; ++i)
        {
            PipelineStage stage = kStageOrder[i];
            m_currentStage.store(stage);

            if (stage == PipelineStage::STAGE3_RENDER)
            {
                // Render's chunk_contributors depend on Optimize's final
                // homographies, so its task list can't be seeded upfront
                // like Ingest/Align/Optimize's (whose task lists only
                // depend on the stable input-image set). Safe/idempotent
                // whether Optimize just ran in this call or was already
                // COMPLETED from a prior session (resume case) -
                // SeedRenderTasks recomputes overlap culling and re-runs
                // SetChunkContributors/CreateTasksIfAbsent, both already
                // idempotent.
                projectManager.SeedRenderTasks();
            }

            std::vector<Task> tasks = projectManager.GetTasksForStage(stage);
            bool alreadyComplete = !tasks.empty() &&
                std::all_of(tasks.begin(), tasks.end(),
                    [](const Task& t) { return t.status == TaskStatus::COMPLETED; });

            if (!alreadyComplete)
            {
                bool ok = scheduler.RunStage(stage, token,
                    [this, i, stage](const Task& task, float stageProgress)
                    {
                        // Executors only ever return a plain bool (see
                        // ITaskExecutor.h) - there's no error string to
                        // surface, but naming exactly which unit gave up
                        // after exhausting its retries is still far more
                        // actionable than the generic "stage failed".
                        if (task.status == TaskStatus::FAILED && m_onLog)
                        {
                            // task.attemptCount is the total attempts made
                            // (TaskScheduler bumps it to the failing count
                            // before this callback fires), not an index.
                            // task.errorMessage is only set by executors
                            // that opt in (see ITaskExecutor.h) - falls
                            // back to nothing rather than a placeholder.
                            m_onLog(L"Task failed permanently: stage=" + std::wstring(StageName(stage)) +
                                L" unitKind=" + Widen(task.unitKind) +
                                L" unitKey=" + Widen(task.unitKey) +
                                L" attempts=" + std::to_wstring(task.attemptCount) +
                                (task.errorMessage.empty() ? L"" : (L" reason=" + Widen(task.errorMessage))));
                        }

                        float overall = (static_cast<float>(i) + stageProgress) / static_cast<float>(kStageCount);
                        m_overallProgress.store(overall);
                        if (m_onProgress) m_onProgress(m_currentStage.load(), overall);
                    });

                if (!ok)
                {
                    m_currentStage.store(token.stop_requested() ? PipelineStage::CANCELLED : PipelineStage::FAILED);
                    if (m_onLog)
                    {
                        m_onLog(token.stop_requested()
                            ? L"Pipeline cancelled during " + std::wstring(StageName(stage)) + L" stage."
                            : L"Pipeline stopped: " + std::wstring(StageName(stage)) + L" stage did not complete.");
                    }
                    return false;
                }
            }

            float overall = static_cast<float>(i + 1) / static_cast<float>(kStageCount);
            m_overallProgress.store(overall);
            if (m_onProgress) m_onProgress(stage, overall);
        }

        m_currentStage.store(PipelineStage::COMPLETED);
        m_overallProgress.store(1.0f);
        if (m_onProgress) m_onProgress(PipelineStage::COMPLETED, 1.0f);
        return true;
    }

    PipelineStage PipelineDriver::GetCurrentStage() const
    {
        return m_currentStage.load();
    }

    float PipelineDriver::GetOverallProgress() const
    {
        return m_overallProgress.load();
    }
}
