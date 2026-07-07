#include "pch.h"
#include "HeaderFiles/PipelineDriver.h"
#include "HeaderFiles/TaskScheduler.h"
#include <algorithm>
#include <iterator>
#include <vector>

namespace WindowsApp::Core
{
    void PipelineDriver::Initialize(ProgressCallback onProgress, LogCallback onLog)
    {
        m_onProgress = std::move(onProgress);
        m_onLog = std::move(onLog);
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

        TaskScheduler scheduler(projectManager);
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
                    [this, i](const Task&, float stageProgress)
                    {
                        float overall = (static_cast<float>(i) + stageProgress) / static_cast<float>(kStageCount);
                        m_overallProgress.store(overall);
                        if (m_onProgress) m_onProgress(m_currentStage.load(), overall);
                    });

                if (!ok)
                {
                    m_currentStage.store(token.stop_requested() ? PipelineStage::CANCELLED : PipelineStage::FAILED);
                    if (m_onLog) m_onLog(L"Stage did not complete; stopping pipeline run.");
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
