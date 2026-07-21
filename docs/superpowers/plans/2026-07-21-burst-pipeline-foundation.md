# Burst Pipeline Foundation (Phase 0)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Phase 0 of `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §8 — lay the
persistence + orchestration foundation for a second pipeline family
(burst-mode: MFNR/HDR+/Night Sight/Super Res Zoom) alongside the existing
panorama pipeline, reusing the `Task`/`TaskScheduler`/`.vfp` machinery from
`docs/ARCHITECTURE.md` §7 rather than inventing new orchestration.
Deliberately **no new kernels, no `IComputeBackend` changes** — this plan
proves the schema/scheduler plumbing generalizes to a second stage sequence,
the same way `2026-07-07-task-scheduler-core.md` proved resume/cancel
against a stub executor before any real GPU work existed for it.

**Depends on:** the shipped v2 foundation (`ProjectManager`'s
`tasks`/`chunk_contributors`/`blob_directory` tables, `TaskScheduler`,
`PipelineDriver`) — all already implemented and in production use by the
panorama pipeline (`docs/superpowers/plans/2026-07-07-*`).

**Architecture:**
- `project_metadata` is already a generic `(key, value)` table (not a fixed-
  column `project` table) — adding `project_type`/`burst_mode` needs no
  schema migration, just two new keys `LoadMetadata` recognizes. This is
  the reason Phase 0 is smaller than it might look: no `ALTER TABLE`, no new
  tables.
- A burst-mode project has no chunk grid (its output is one merged frame,
  not a tiled gigapixel canvas) — `CreateBurstProject` is a new entry point
  that skips the chunk-grid population loop `CreateProject` runs, rather
  than overloading `CreateProject` with meaningless `totalWidth`/
  `totalHeight`/`chunkSize` arguments for a mode that doesn't chunk.
- `PipelineDriver::Run`'s stage sequence becomes derived from
  `ProjectManager::GetProjectType()` instead of hardcoded
  `STAGE0_INGEST..STAGE3_RENDER` — this is the one behavior change to
  existing code. `MainWindow.xaml.cpp`'s call site is unaffected (no
  signature change), since a loaded panorama project reports
  `ProjectType::PANORAMA` and gets exactly today's sequence.
- Per `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §3, `BURST_MERGE`'s *executor*
  differs by `BurstMode` (MFNR's robust merge vs. HDR+'s FFT/Wiener merge
  vs. Super-Res/Night-Sight's kernel-regression merge are different
  algorithms) — this plan only adds the stage/mode plumbing, not any
  executor. A single `ITaskExecutor` per `BURST_*` stage is registered by
  the composition root same as today's four stages; picking the right
  merge algorithm *within* `BURST_MERGE` by `BurstMode` is later-plan scope.

**Tech Stack:** C++20, existing `WindowsApp.Core`/`WindowsApp.Tests`
structure, plus `tests/engine_smoke` (CMake/ctest, cross-platform —
verifiable on Linux, unlike `WindowsApp.Tests`).

## Global Constraints

- No `WindowsApp.Compute`/`WindowsApp.Vulkan` changes, no UI changes beyond
  what's needed to keep `MainWindow.xaml.cpp` compiling unchanged.
- Do not touch the `chunks`/`chunk_contributors` tables or panorama's
  existing `Seed*Tasks` methods.
- `PipelineStage`'s existing values (`STAGE0_INGEST..STAGE3_RENDER`,
  `COMPLETED`/`CANCELLED`/`FAILED`) keep their exact names and meaning —
  only additive changes.

---

### Task 1: `PipelineStage` additions + `ProjectType`/`BurstMode`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/Types.h`

**Interfaces:**
- Produces: `PipelineStage::BURST_ALIGN`/`BURST_MERGE`/`BURST_FINISH`,
  `enum class ProjectType { PANORAMA, BURST }`,
  `enum class BurstMode { NONE, MFNR, HDR_PLUS, NIGHT_SIGHT, SUPER_RES }`.

- [ ] **Step 1:** Add the three `BURST_*` values to `PipelineStage`, after
  `STAGE3_RENDER` and before the shared terminal states
  (`COMPLETED`/`CANCELLED`/`FAILED`).
- [ ] **Step 2:** Add `ProjectType` and `BurstMode` enums near
  `PipelineStage`. `BurstMode::NONE` is the value a `PANORAMA` project
  reports (not a 5th "no mode" burst mode) — `ProjectType`/`BurstMode` are
  deliberately two separate enums rather than one 5-way enum, so a future
  panorama-specific field never has to special-case "which of these 5
  values are actually panorama."
- [ ] **Step 3: Parse check** —
  `grep -n "BURST_ALIGN\|BURST_MERGE\|BURST_FINISH\|enum class ProjectType\|enum class BurstMode" WindowsApp.Core/HeaderFiles/Types.h`
  — expect every name present.

---

### Task 2: `ProjectManager` burst project type + metadata round-trip

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
- Produces:
  ```cpp
  bool CreateBurstProject(const std::wstring& dbPath, BurstMode mode);
  ProjectType GetProjectType() const;
  BurstMode GetBurstMode() const;
  ```

- [ ] **Step 1:** Add `m_projectType`/`m_burstMode` members, defaulted to
  `ProjectType::PANORAMA`/`BurstMode::NONE` (so every existing panorama
  project — no `project_type` key in its `project_metadata` table — loads
  as `PANORAMA` with no migration needed).
- [ ] **Step 2:** Implement `CreateBurstProject`: same `sqlite3_open` +
  `PRAGMA` + schema-creation prologue as `CreateProject` (factor the shared
  part into a private helper rather than duplicating the schema string —
  the two `CREATE TABLE IF NOT EXISTS` blocks in `CreateProject`/
  `LoadProject` already show this project tolerates some duplication, but a
  third copy is where it stops being worth it), skip the chunk-grid loop
  entirely, and write `project_type='BURST'` + `burst_mode=<mode>` into
  `project_metadata`. `totalWidth`/`totalHeight` stay `0` for a burst
  project (meaningless until a real burst executor sets them from the
  first ingested frame — out of scope here).
- [ ] **Step 3:** `CreateProject` (panorama) writes `project_type='PANORAMA'`
  (and no `burst_mode` key, or `burst_mode='NONE'` — pick one and keep it
  consistent with `LoadMetadata`'s default) into `project_metadata`
  alongside its existing `total_width`/`total_height` keys.
- [ ] **Step 4:** `LoadMetadata` gains `else if (k == "project_type")`/
  `else if (k == "burst_mode")` branches parsing into the new members
  (string enums, same pattern as `ToString(PipelineStage)`/
  `ParsePipelineStage` in the anonymous namespace — add
  `ToString(ProjectType)`/`ParseProjectType`/`ToString(BurstMode)`/
  `ParseBurstMode` alongside them). Missing key (older/panorama project) ⇒
  keeps the constructor defaults from Step 1.
- [ ] **Step 5:** Add the two accessors (trivial member returns).
- [ ] **Step 6: Header/source consistency check** —
  `grep -n "CreateBurstProject\|GetProjectType\|GetBurstMode" WindowsApp.Core/HeaderFiles/ProjectManager.h WindowsApp.Core/SourceFiles/ProjectManager.cpp`
  — expect declarations and matching definitions.

---

### Task 3: `PipelineDriver` stage-sequence generalization

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/PipelineDriver.h`
- Modify: `WindowsApp.Core/SourceFiles/PipelineDriver.cpp`

**Interfaces:**
- `Run`'s signature is unchanged (`bool Run(ProjectManager&,
  CancellationToken)`) — the sequence it drives becomes internal, derived
  from `projectManager.GetProjectType()`, instead of a hardcoded local
  array/switch.

- [ ] **Step 1:** Replace the hardcoded
  `{STAGE0_INGEST, STAGE1_ALIGN, STAGE2_OPTIMIZE, STAGE3_RENDER}` sequence
  `Run` iterates with a small local function/table returning that array for
  `ProjectType::PANORAMA` and `{BURST_ALIGN, BURST_MERGE, BURST_FINISH}`
  for `ProjectType::BURST`. Everything else about `Run`'s loop (skip stages
  whose tasks are already fully `COMPLETED`, stop at the first stage that
  returns `false` from `RunStage`, progress/log callbacks) is unchanged —
  this is purely "which stages," not "how a stage runs."
  - Note the render-seeding special case already wired in `Run` (Render's
    `chunk_contributors` seeding happens only after Optimize, per
    `2026-07-07-real-project-creation-ui.md` Task 1) is
    `PipelineStage::STAGE2_OPTIMIZE`/`STAGE3_RENDER`-specific — guard it so
    it only fires for the panorama sequence, not for burst mode where those
    stage values never appear in the loop.
- [ ] **Step 2: Header/source consistency check** — confirm `Run`'s
  signature in the header is untouched
  (`grep -n "bool Run" WindowsApp.Core/HeaderFiles/PipelineDriver.h`
  shows exactly the one, unchanged declaration) and that
  `PipelineDriver.cpp` now references `BURST_ALIGN`/`BURST_MERGE`/
  `BURST_FINISH` somewhere.

---

### Task 4: Cross-platform test coverage (`tests/engine_smoke`)

**Files:**
- Modify: `tests/engine_smoke/main.cpp`

**Interfaces:**
- Consumes: `ProjectManager::CreateBurstProject`/`GetProjectType`/
  `GetBurstMode`, `PipelineDriver`, a local stub `ITaskExecutor` (this test
  binary has no existing stub executor to reuse — `StubTaskExecutor.h`
  lives under `WindowsApp.Tests`, MSBuild-only — so add a small
  self-contained one here, mirroring its shape).

- [ ] **Step 1: Burst project type round-trip** — `CreateBurstProject` with
  `BurstMode::MFNR` at a temp path, assert `GetProjectType() ==
  ProjectType::BURST` and `GetBurstMode() == BurstMode::MFNR` immediately
  after create, then `LoadProject` that same path fresh and assert both
  again (proves the `project_metadata` round-trip survives a real
  close+reopen, not just the in-memory value set at create time).
- [ ] **Step 2: Panorama default stays default** — `CreateProject` (the
  existing panorama entry point) and assert `GetProjectType() ==
  ProjectType::PANORAMA`, `GetBurstMode() == BurstMode::NONE` — the
  no-migration-needed claim from this plan's Architecture note, checked by
  a real test rather than just asserted in prose.
- [ ] **Step 3: Burst stage sequence runs end-to-end** — on a
  `CreateBurstProject` project, seed one task per `BURST_ALIGN`/
  `BURST_MERGE`/`BURST_FINISH` via the existing `CreateTasksIfAbsent`
  (`unit_kind="frame"`, arbitrary `unit_key`), register the local stub
  executor for all three stages on a `PipelineDriver`, call `Run` with a
  never-cancelled token, assert it returns `true` and every seeded task is
  `COMPLETED` in that order (stub can record stage-entry order to check
  `BURST_ALIGN` ran before `BURST_MERGE` before `BURST_FINISH`).
- [ ] **Step 4: Existing panorama scenarios still pass unmodified** — no
  new assertions needed here, just confirm `ctest` still runs the file's
  existing scenarios (`RunCrossTierSimdKernelChecks`, the SQLite/
  StorageEngine round-trip) after this file's edits — a regression, not a
  new-feature check.
- [ ] **Step 5: Build + run** —
  ```bash
  cmake --build build --target engine_smoke_tests
  ctest --test-dir build -R engine_smoke --output-on-failure
  ```
  Expected: pass, including the three new scenarios from Steps 1-3.

---

### Task 5: Verify nothing else regresses

**Files:** none (verification only)

- [ ] **Step 1: Full Linux CMake/ctest suite** —
  ```bash
  cmake --build build
  ctest --test-dir build --output-on-failure
  ```
  Expected: all existing targets (`pipeline_e2e`, `render_blend`,
  `raw_probe` where applicable) still pass — this plan's changes are
  additive to `Types.h`/`ProjectManager`/`PipelineDriver`, all three
  contain the shared panorama codepath, so a real full-suite run is the
  only real evidence "additive" is true and not just intended.
- [ ] **Step 2: Windows MSBuild + `WindowsApp.Tests`** — via the
  `windows-host` MCP tools against `win-thanh` (`sync_dir` → `run_command`
  a `dotnet msbuild WindowsApp.slnx` build → run `WindowsApp.Tests` through
  `vstest.console.exe`), per this repo's established autonomous-build
  convention (prefer driving the real Windows box over asking the user to
  build). Confirms `ProjectManagerTaskTests.cpp`/`PipelineDriverTests.cpp`
  (MSBuild-only, not exercised by Step 1) still pass and that
  `MainWindow.xaml.cpp`'s existing `PipelineDriver::Run` call site still
  compiles unchanged against the new signature-preserving internals.

## Self-Review

- Spec coverage: Phase 0 from `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §8 (schema
  additions, `TaskScheduler`/`PipelineDriver` generalized for a second
  pipeline family, no new kernels) is fully covered by Tasks 1-4; Task 5 is
  the "did it actually stay additive" check called out as a risk in the
  Architecture note.
- Placeholder scan: no placeholder steps remain.
- Type consistency: `ProjectType`/`BurstMode`/`PipelineStage` names match
  across `Types.h`, `ProjectManager.h`/`.cpp`, `PipelineDriver.cpp`, and the
  test file.
