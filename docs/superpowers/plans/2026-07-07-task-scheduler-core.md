# TaskScheduler & PipelineDriver (Stub-Executor Slice)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the generic, stage-agnostic `ITaskExecutor` /
`TaskScheduler` / `PipelineDriver` trio from `docs/ARCHITECTURE.md` §14.1,
proven out against a **stub executor** rather than real GPU/DirectStorage
work. This deliberately separates two different kinds of risk: "is the
resume/cancel/reclaim state machine correct" (testable today, in plain
C++, with no CUDA) versus "does the real GPU pipeline work" (later plans).
Getting the first one right and thoroughly tested here means later plans
that plug in real executors (RawIngest, Align, Optimize, Render) inherit
correct cancellation and resume for free instead of each having to get it
right independently.

**Depends on:** `2026-07-07-vfp-project-schema.md` (needs `Task`,
`TaskStatus`, `PipelineStage`, `CancellationToken`, and `ProjectManager`'s
task CRUD).

**Architecture:** `TaskScheduler` owns no domain knowledge about what a
task *does* — it only knows how to look up rows via `ProjectManager` and
dispatch them to whichever `ITaskExecutor` is registered for that stage.
The in-flight window in this slice is implemented with `std::async`
(`std::launch::async`) capped at a small constant, **not** CUDA
streams/DirectStorage queues — later plans replace the internals of
"how a dispatched task overlaps with the next" without changing
`TaskScheduler`'s public contract (`RunStage`), which is the point of
proving the contract out against something simple first.

**Tech Stack:** C++20 (`std::stop_token`, `std::jthread`, `std::async`,
`std::future`), existing `WindowsApp.Core`/`WindowsApp.Tests` structure.

## Global Constraints

- No CUDA, no DirectStorage, no UI changes in this plan.
- `ITaskExecutor` implementations for real stages (RawIngest, Align, ...)
  are explicitly out of scope — only a test-double `StubTaskExecutor` is
  implemented here.
- Cancellation must only be checked *between* task dispatches, never used
  to abort an in-flight `std::future` — matches `docs/ARCHITECTURE.md`
  §7.2's "in-flight work always finishes and commits" rule, and this
  matters even with the simplified `std::async` stand-in because it's the
  behavior later CUDA-graph-backed executors must also honor.

---

### Task 1: `ITaskExecutor` interface

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/ITaskExecutor.h`

**Interfaces:**
- Produces: `WindowsApp::Core::ITaskExecutor`

- [ ] **Step 1: Declare the interface**

```cpp
#pragma once
#include "Types.h"

namespace WindowsApp::Core
{
    class ITaskExecutor
    {
    public:
        virtual ~ITaskExecutor() = default;

        // Contract: must be idempotent — re-invoked with the same Task
        // after a crash must be safe and produce the same committed
        // result. Expected failures (bad input, transient I/O error)
        // return false and should set an error message on the task's
        // owning ProjectManager row via the caller; exceptions are
        // reserved for programmer errors, not runtime conditions.
        virtual bool Execute(Task& task, CancellationToken token) = 0;
    };
}
```

- [ ] **Step 2: Add to `WindowsApp.Core.vcxproj`**

Add the new header to the `<ItemGroup>` of `ClInclude` entries.

---

### Task 2: `TaskScheduler`

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/TaskScheduler.h`
- Create: `WindowsApp.Core/SourceFiles/TaskScheduler.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
- Consumes: `ProjectManager` (Plan 1), `ITaskExecutor` (Task 1).
- Produces:
  ```cpp
  class TaskScheduler
  {
  public:
      TaskScheduler(ProjectManager& projectManager);

      void RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor);

      // Runs every task for `stage` not already COMPLETED. Returns false
      // if the stage did not fully complete (cancelled, or a task
      // exhausted its retry budget). Never throws for expected failure.
      bool RunStage(PipelineStage stage, CancellationToken token,
                    std::function<void(const Task&, float stageProgress)> onTaskProgress);

  private:
      ProjectManager& m_projectManager;
      std::unordered_map<PipelineStage, std::shared_ptr<ITaskExecutor>> m_executors;
      static constexpr size_t kMaxInFlight = 4;
      static constexpr int kMaxAttempts = 3;
  };
  ```

- [ ] **Step 1: Implement `RunStage`'s reclaim + dispatch loop**

1. Call `m_projectManager.ReclaimStaleRunningTasks(stage)` first — every
   call to `RunStage`, not just the first one for a project, so resuming
   mid-stage after a crash is also safe (matches `docs/ARCHITECTURE.md`
   §7.2 — this is called per-`RunStage`, not just once at project load).
2. Fetch `GetTasksForStage(stage)`, filter to non-`COMPLETED`.
3. Maintain up to `kMaxInFlight` outstanding `std::future<bool>` (via
   `std::async(std::launch::async, ...)`), each wrapping: mark task
   `RUNNING` → call the registered executor's `Execute` → on success,
   `CommitTaskOutput`/`UpdateTaskStatus(COMPLETED)`; on failure, increment
   `attempt_count` and either requeue (`< kMaxAttempts`) or mark `FAILED`.
4. Before dispatching each *new* task, check `token.stop_requested()`; if
   set, stop dispatching but drain (await) all currently in-flight futures
   to completion before returning — never abandon an in-flight future.
5. Invoke `onTaskProgress` after each task settles (completed or failed),
   with `stageProgress = completedCount / totalCount`.
6. Return `true` only if every task for the stage ended `COMPLETED`;
   `false` if cancelled or any task hit `kMaxAttempts` and is `FAILED`.

- [ ] **Step 2: Header/source consistency check**

Run:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\TaskScheduler.h,WindowsApp.Core\SourceFiles\TaskScheduler.cpp -Pattern 'RunStage|RegisterExecutor|ReclaimStaleRunningTasks'
```
Expected: declarations in the header, matching definitions in the source.

---

### Task 3: `PipelineDriver`

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/PipelineDriver.h`
- Create: `WindowsApp.Core/SourceFiles/PipelineDriver.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
- Consumes: `TaskScheduler` (Task 2), `ProjectManager` (Plan 1).
- Produces:
  ```cpp
  class PipelineDriver
  {
  public:
      using ProgressCallback = std::function<void(PipelineStage stage, float overallProgress)>;
      using LogCallback = std::function<void(const std::wstring&)>;

      void Initialize(ProgressCallback onProgress, LogCallback onLog);
      void RegisterExecutor(PipelineStage stage, std::shared_ptr<ITaskExecutor> executor);

      // Drives STAGE0_INGEST -> STAGE1_ALIGN -> STAGE2_OPTIMIZE ->
      // STAGE3_RENDER in order, skipping stages whose tasks are already
      // fully COMPLETED. Safe to call again after a previous cancelled
      // run or a crash — every call is a resume.
      bool Run(ProjectManager& projectManager, CancellationToken token);

      PipelineStage GetCurrentStage() const;
      float GetOverallProgress() const;

  private:
      TaskScheduler m_scheduler; // constructed against the ProjectManager passed to Run
      ProgressCallback m_onProgress;
      LogCallback m_onLog;
      std::atomic<PipelineStage> m_currentStage{ PipelineStage::IDLE };
      std::atomic<float> m_overallProgress{ 0.0f };
  };
  ```

  Deliberately absent: a `Cancel()` method. Per `docs/ARCHITECTURE.md`
  §14.1, cancellation is caller-owned — whoever calls `Run` owns the
  `std::stop_source` and passes only the `token` half.

- [ ] **Step 1: Implement the stage loop**

For each stage in order: skip immediately (no scheduler call at all) if
`GetTasksForStage(stage)` is non-empty and every task is already
`COMPLETED`; otherwise call `m_scheduler.RunStage(stage, token, ...)` and
stop the whole `Run` (return `false`) the moment one stage returns `false`.
Update `m_currentStage`/`m_overallProgress` (stage index / 4 as a coarse
weight is sufficient for this slice — no need to match Stage 3's `0.30 +
progress*0.70` weighting from v1 yet, that's a later-plan refinement) and
invoke `m_onProgress` as stages advance.

- [ ] **Step 2: Header/source consistency check**

Run:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\PipelineDriver.h,WindowsApp.Core\SourceFiles\PipelineDriver.cpp -Pattern 'PipelineDriver::Run|GetCurrentStage|GetOverallProgress'
```
Expected: declarations in the header, matching definitions in the source.
Also confirm no `Cancel()` method was added:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\PipelineDriver.h -Pattern 'Cancel'
```
Expected: no matches.

---

### Task 4: `StubTaskExecutor` test double

**Files:**
- Create: `WindowsApp.Tests/StubTaskExecutor.h`

**Interfaces:**
- Implements: `ITaskExecutor`

- [ ] **Step 1: Implement a configurable stub**

Simulates variable-duration async work (`std::this_thread::sleep_for` a
small configurable duration) and honors cancellation only at its own entry
point (checks `token.stop_requested()` once at the start of `Execute`, then
runs to completion regardless — modeling "a dispatched unit of GPU work
always finishes once started," per this plan's Global Constraints). Should
support a test-controlled failure mode (fail the first N attempts for a
given `unitKey`, to exercise `TaskScheduler`'s retry path) and a
completion counter the test can assert against.

---

### Task 5: Test coverage

**Files:**
- Create: `WindowsApp.Tests/PipelineDriverTests.cpp`
- Modify: `WindowsApp.Tests/WindowsApp.Tests.vcxproj`

**Interfaces:**
- Consumes: `PipelineDriver`, `TaskScheduler`, `StubTaskExecutor`.

- [ ] **Step 1: Fresh-run test**

Create a project, seed a handful of `Task` rows across two stages via
`CreateTasksIfAbsent`, register `StubTaskExecutor` for both, call
`PipelineDriver::Run` with a never-cancelled token. Assert it returns
`true` and every task is `COMPLETED`.

- [ ] **Step 2: Resume-skips-completed test**

Same setup; manually `CommitTaskOutput` half the tasks in stage 1 before
calling `Run`. Assert the stub executor's completion counter only
increments for the *not*-already-completed tasks (proves resume doesn't
redo finished work).

- [ ] **Step 3: Crash-simulation test**

Manually set one task to `RUNNING` (simulating a prior process that died
mid-task) before calling `Run`. Assert that task is reclaimed and
re-executed (via the stub's completion counter) and ends `COMPLETED`.

- [ ] **Step 4: Cancellation test**

Seed enough tasks that `kMaxInFlight` is exceeded, use a `StubTaskExecutor`
configured with a short artificial delay, and call `request_stop()` on the
`std::stop_source` shortly after starting `Run` on a separate thread.
Assert: `Run` returns `false`; tasks that were already in-flight at the
time of the stop request end `COMPLETED` (never left `RUNNING` or
half-done); tasks never dispatched remain `PENDING` (not `CANCELLED` —
this slice doesn't need per-task `CANCELLED` marking, only "stopped
dispatching new work").

- [ ] **Step 5: Try available build**

Run:
```powershell
dotnet msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64 /v:minimal
```
Expected in this shell: may fail because Visual Studio C++/CUDA toolchains
are unavailable. If it fails only for missing targets/toolset, report that
and ask the user to build + run `WindowsApp.Tests` in Visual Studio Test
Explorer.

## Self-Review

- Spec coverage: `TaskScheduler`/`PipelineDriver` contracts from
  `docs/ARCHITECTURE.md` §14.1 are implemented and independently testable
  without any GPU dependency; resume and cancellation are both covered by
  explicit tests rather than asserted only by design-doc prose.
- Placeholder scan: no placeholder steps remain; `StubTaskExecutor` is a
  deliberate, documented test double, not a placeholder for missing
  production code.
- Type consistency: `ITaskExecutor`/`TaskScheduler`/`PipelineDriver` names
  and signatures match `docs/ARCHITECTURE.md` §14.1 exactly, so later plans
  plugging in real executors don't need to touch this plan's public API.
