# .vfp Project Schema & Task Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lay the persistence foundation for the v2 architecture
(`docs/ARCHITECTURE.md` §7, §9): a unified `tasks` table that every future
pipeline stage schedules work through, plus `chunk_contributors` and
`blob_directory` tables. This plan touches `WindowsApp.Core` only — no
CUDA, no DirectStorage, no UI. It also retires the v1 orchestration classes
(`WorkflowController`, `CacheManager`) that this and later plans replace,
since this is a ground-up redesign, not an extension of v1 (per
`docs/ARCHITECTURE.md`'s framing and the user's explicit "we can restart"
direction).

**Architecture:** Extends `ProjectManager`'s existing SQLite-C-API style
(`sqlite3_exec`/`sqlite3_prepare_v2`, tables created in `CreateProject`,
loaded in `LoadProject`) rather than introducing a new persistence
abstraction — matches the existing `input_images`/`chunks` pattern. One
deliberate deviation: the existing `AddInputImage`/`UpdateChunkStatus`
build SQL via `snprintf` string concatenation. The new Task/Blob methods use
`sqlite3_bind_*` prepared statements instead, since `unit_key` values are
derived from file paths and other data-driven strings that aren't safe to
concatenate — this is new code, not a refactor of the existing methods, so
it can just be done correctly rather than replicating the old pattern.

Tasks are not eagerly cached into a member vector the way `m_chunks`/
`m_inputImages` are — a project can have thousands of task rows (one per
image/pair/chunk across every stage) and only the ones a given stage cares
about are ever needed at once, so `GetTasksForStage` is a live query, not a
cache lookup.

**Tech Stack:** C++20, vendored SQLite (`WindowsApp.Core/sqlite3`), existing
MSBuild/vcxproj structure.

## Global Constraints

- Do not touch `WindowsApp.Compute`, `WindowsApp/WindowsApp` (UI), or
  `WindowsApp.Core/libraw`/`sqlite3` in this plan.
- `PipelineStage` moves from `WorkflowController.h` (being deleted) to
  `Types.h` — do not leave two definitions of it alive at once anywhere in
  the tree.
- Full build may require Visual Studio C++/CUDA toolchains not available to
  `dotnet msbuild` in a plain shell — verification steps note this.

---

### Task 1: Retire v1 orchestration classes

**Files:**
- Delete: `WindowsApp.Core/HeaderFiles/WorkflowController.h`
- Delete: `WindowsApp.Core/SourceFiles/WorkflowController.cpp`
- Delete: `WindowsApp.Core/HeaderFiles/CacheManager.h`
- Delete: `WindowsApp.Core/SourceFiles/CacheManager.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj` (and `.vcxproj.filters`
  if present)

**Interfaces:**
- Removes: `WindowsApp::Core::WorkflowController`, `::CacheManager`, and the
  `PipelineStage` enum as currently declared in `WorkflowController.h`
  (recreated in Task 2, in `Types.h`).

- [ ] **Step 1: Confirm nothing else references these classes**

Run:
```powershell
Select-String -Path WindowsApp\WindowsApp\**\*.cpp,WindowsApp\WindowsApp\**\*.h -Pattern 'WorkflowController|CacheManager' -ErrorAction SilentlyContinue
```
Expected: no matches — the UI project has no `ProjectReference` to
`WindowsApp.Core` yet (confirmed in `docs/ARCHITECTURE.md` §4), so nothing
outside `WindowsApp.Core`/`WindowsApp.Tests` can reference these types. If
`WindowsApp.Tests/UnitTests.cpp` references either class beyond the current
placeholder test, stop and re-scope this task before deleting.

- [ ] **Step 2: Delete the four files and remove their vcxproj entries**

Remove the corresponding `<ClInclude>`/`<ClCompile>` entries from
`WindowsApp.Core.vcxproj` (and matching `<Filter>` entries in
`WindowsApp.Core.vcxproj.filters` if that file exists).

- [ ] **Step 3: Confirm the project file no longer lists the deleted files**

Run:
```powershell
Select-String -Path WindowsApp.Core\WindowsApp.Core.vcxproj -Pattern 'WorkflowController|CacheManager'
```
Expected: no matches.

---

### Task 2: Shared v2 types

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/Types.h`

**Interfaces:**
- Produces: `PipelineStage` (relocated), `TaskStatus`, `Task`,
  `BlobDirectoryEntry`, `CancellationToken` alias.

- [ ] **Step 1: Add `PipelineStage` to `Types.h`**

Same members as the enum being deleted with `WorkflowController.h`:
`IDLE, STAGE0_INGEST, STAGE1_ALIGN, STAGE2_OPTIMIZE, STAGE3_RENDER,
COMPLETED, CANCELLED, FAILED` — adds `STAGE0_INGEST`, which v1 never had
(RawIngest is a new stage per `docs/ARCHITECTURE.md` §4.1).

- [ ] **Step 2: Add `TaskStatus` and `Task`**

```cpp
enum class TaskStatus { PENDING, RUNNING, COMPLETED, FAILED, CANCELLED };

struct Task
{
    int64_t taskId = 0;
    PipelineStage stage = PipelineStage::IDLE;
    std::string unitKind;   // "image" | "image_band" | "pair" | "chunk" | "ba_checkpoint"
    std::string unitKey;    // e.g. "img_7", "img_7:band_3", "img_2:img_9", "C_4_2"
    TaskStatus status = TaskStatus::PENDING;
    int attemptCount = 0;
    std::optional<int64_t> outputBlobId;
    std::string checkpointJson;
};
```

- [ ] **Step 3: Add `BlobDirectoryEntry`**

```cpp
struct BlobDirectoryEntry
{
    int64_t blobId = 0;
    std::wstring shardFile;
    int64_t offset = 0;
    int64_t length = 0;
    std::optional<int64_t> compressedLength;
    std::string formatTag; // e.g. "raw_rgb48", "gdeflate", "nvjpeg"
};
```

- [ ] **Step 4: Add the `CancellationToken` alias**

```cpp
using CancellationToken = std::stop_token;
```

Placed here (not in a new header) because it's used by `Task`-adjacent
signatures across both `WindowsApp.Core` and, later, `WindowsApp.Compute`.

- [ ] **Step 5: Parse check**

Run:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\Types.h -Pattern 'PipelineStage|TaskStatus|struct Task|BlobDirectoryEntry|CancellationToken'
```
Expected: each name appears.

---

### Task 3: Schema — `tasks`, `chunk_contributors`, `blob_directory`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
- Consumes: `Task`, `TaskStatus`, `BlobDirectoryEntry`, `PipelineStage` from
  Task 2.
- Produces: three new tables, created in both `CreateProject` (fresh
  project) and `LoadProject` (via `CREATE TABLE IF NOT EXISTS`, so opening
  an older `.vfp` file — including one created earlier in this same
  migration — never fails for a missing table).

- [ ] **Step 1: Add the schema block**

Extend the `schema` string in `CreateProject` (and add an equivalent
`CREATE TABLE IF NOT EXISTS` block that `LoadProject` runs right after
`sqlite3_open`, before `LoadMetadata`/`LoadInputImages`/`LoadChunks`):

```sql
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
```

The `UNIQUE(stage, unit_kind, unit_key)` constraint is load-bearing: it's
what makes re-seeding a stage's task list on a resumed run a no-op for
already-created rows (`INSERT OR IGNORE`) instead of creating duplicates.

- [ ] **Step 2: Confirm schema is present after both create and load paths**

Run:
```powershell
Select-String -Path WindowsApp.Core\SourceFiles\ProjectManager.cpp -Pattern 'CREATE TABLE IF NOT EXISTS tasks|CREATE TABLE IF NOT EXISTS chunk_contributors|CREATE TABLE IF NOT EXISTS blob_directory'
```
Expected: each string appears at least twice (once for `CreateProject`'s
inline schema, once for the block `LoadProject` runs).

---

### Task 4: Task / contributor / blob CRUD on `ProjectManager`

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`

**Interfaces:**
- Produces:
  ```cpp
  // Tasks
  bool CreateTasksIfAbsent(const std::vector<Task>& tasks); // INSERT OR IGNORE, keyed by the UNIQUE constraint
  std::vector<Task> GetTasksForStage(PipelineStage stage) const;
  bool UpdateTaskStatus(int64_t taskId, TaskStatus status);
  bool CommitTaskOutput(int64_t taskId, int64_t outputBlobId); // sets status=COMPLETED and output_blob_id together
  bool UpdateTaskCheckpoint(int64_t taskId, const std::string& checkpointJson);
  int  ReclaimStaleRunningTasks(PipelineStage stage); // UPDATE ... SET status='PENDING' WHERE status='RUNNING'; returns row count

  // Chunk contributors
  bool SetChunkContributors(const std::string& chunkId, const std::vector<int>& imageIds);
  std::vector<int> GetChunkContributors(const std::string& chunkId) const;

  // Blob directory
  int64_t AddBlobDirectoryEntry(const BlobDirectoryEntry& entry); // returns new blobId
  std::optional<BlobDirectoryEntry> GetBlobDirectoryEntry(int64_t blobId) const;
  ```

- [ ] **Step 1: Implement Task CRUD using `sqlite3_bind_*` prepared statements**

Not `snprintf` — see this plan's Architecture note on why the new code
diverges from `AddInputImage`'s existing string-concatenation style.
`CommitTaskOutput` must set `status` and `output_blob_id` in the same
`UPDATE` statement (single row write), matching `docs/ARCHITECTURE.md` §5's
durability ordering: the blob's bytes and fence-completion happen before
this call is made by the caller, not enforced here — this method is just
the atomic metadata commit half of that ordering.

- [ ] **Step 2: Implement `ReclaimStaleRunningTasks`**

```sql
UPDATE tasks SET status = 'PENDING' WHERE stage = ? AND status = 'RUNNING';
```
Returns `sqlite3_changes(m_db)` after the update so callers (Plan 2's
`TaskScheduler`) can log how many stale tasks were reclaimed.

- [ ] **Step 3: Implement chunk-contributor and blob-directory CRUD**

`SetChunkContributors` should replace any existing rows for that
`chunk_id` (delete-then-insert inside one call) so it's safe to call
repeatedly if overlap culling is ever recomputed.

- [ ] **Step 4: Update `CloseProject`**

No new in-memory caches were added (per this plan's Architecture note), so
`CloseProject` needs no changes beyond what Task 1's deletions already
require removing (any `CacheManager`/`WorkflowController` member, if
`ProjectManager` held one — confirm it doesn't; it shouldn't, based on the
current header).

- [ ] **Step 5: Header/source consistency check**

Run:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\ProjectManager.h -Pattern 'CreateTasksIfAbsent|GetTasksForStage|UpdateTaskStatus|CommitTaskOutput|ReclaimStaleRunningTasks|SetChunkContributors|GetChunkContributors|AddBlobDirectoryEntry|GetBlobDirectoryEntry'
```
Expected: each name appears in the header, and a matching definition
exists in `ProjectManager.cpp`.

---

### Task 5: Test coverage

**Files:**
- Modify: `WindowsApp.Tests/UnitTests.cpp` (or add
  `WindowsApp.Tests/ProjectManagerTaskTests.cpp` — prefer a new file to
  keep the placeholder test class separate from real coverage)
- Modify: `WindowsApp.Tests/WindowsApp.Tests.vcxproj` if a new file is added

**Interfaces:**
- Consumes: `ProjectManager`'s new methods from Task 4.

- [ ] **Step 1: Schema smoke test**

`CreateProject` on a temp path, assert `GetTasksForStage` /
`GetChunkContributors` / `GetBlobDirectoryEntry` all succeed and return
empty results on a fresh project (proves the tables exist and are
queryable, not just that `CREATE TABLE` didn't throw).

- [ ] **Step 2: Task CRUD round-trip test**

`CreateTasksIfAbsent` with a few tasks for one stage, `GetTasksForStage`
returns them with `PENDING` status; `CommitTaskOutput` on one, re-query,
assert `status == COMPLETED` and `outputBlobId` matches.

- [ ] **Step 3: `UNIQUE` re-seed test**

Call `CreateTasksIfAbsent` twice with the same `(stage, unitKind, unitKey)`
rows; assert `GetTasksForStage` still returns exactly one row per unit
(proves re-seeding on a resumed run doesn't duplicate).

- [ ] **Step 4: Crash-simulation / stale-`RUNNING` test**

`UpdateTaskStatus` a task to `RUNNING` (simulating a task that started but
never committed), then call `ReclaimStaleRunningTasks` for its stage;
assert it returns `1` and the task is back to `PENDING`. This is the
concrete test for `docs/ARCHITECTURE.md` §7.2's crash-recovery rule.

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

- Spec coverage: schema, types, CRUD, and crash-recovery behavior from
  `docs/ARCHITECTURE.md` §7/§9 are all covered; v1's dead orchestration
  classes are removed rather than left to rot alongside the new code.
- Placeholder scan: no placeholder steps remain.
- Type consistency: `Task`/`TaskStatus`/`BlobDirectoryEntry` names match
  across `Types.h`, `ProjectManager.h`/`.cpp`, and the test file.
