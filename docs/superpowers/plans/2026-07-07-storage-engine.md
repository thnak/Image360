# StorageEngine — Sharded Blob Container (WriteFile Path)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `WindowsApp::Core::StorageEngine` — the sharded
`<project>.NNNN.vfpdata` blob container from `docs/ARCHITECTURE.md` §5,
wired to `ProjectManager`'s `blob_directory` table — and establish the
write-before-commit durability ordering that `ProjectManager::CommitTaskOutput`'s
caller must uphold (§5, §7.2).

**Scope cut (deliberate):** this plan ships the **WriteFile-based path
only** — plain `CreateFileW`/`WriteFile`/`ReadFile`, no DirectStorage, no
GDeflate compression, no CUDA↔D3D12 interop. `docs/ARCHITECTURE.md` §5 and
§13 flag DirectStorage's write path as less mature than its read path and
say to "start on conventional overlapped `WriteFile` and move to
DirectStorage writes once verified stable" — this plan is that starting
point for *both* directions (not just writes), so the real DirectStorage
integration (a separate, later plan) swaps `StorageEngine`'s internals
without changing its public contract, the same way `TaskScheduler` was
proven against a stub executor before any GPU executor existed. Blobs are
stored uncompressed (`blob_directory.compressed_length` stays `NULL`) —
GDeflate is a DirectStorage GPU-decode format and isn't available outside
that path.

**Depends on:** `2026-07-07-vfp-project-schema.md` (needs `BlobDirectoryEntry`,
`ProjectManager::AddBlobDirectoryEntry`/`GetBlobDirectoryEntry`).

**Architecture:** One `StorageEngine` per open project, living alongside
its `ProjectManager` (both owned by whatever opens a `.vfp` file — later,
`PipelineDriver`/`MainWindow`). It manages an "active" shard file for
appends and rolls to a new shard past a size threshold. Shard filenames
are stored as bare filenames (not full paths) in `blob_directory.shard_file`,
resolved relative to the project directory at open time — so a project
folder can be moved/copied without invalidating blob references.

**Tech Stack:** Win32 file I/O (`CreateFileW`, `WriteFile`, `ReadFile`,
`SetFilePointerEx`, `GetFileSizeEx`), existing `WindowsApp.Core::ProjectManager`/
`Types.h::PixelBuffer`. No new NuGet/SDK dependencies for this plan.

## Global Constraints

- No DirectStorage, no CUDA, no compression in this plan — see the scope
  cut above. A later plan swaps the internals; the public `StorageEngine`
  contract (`Open`/`Close`/`WriteBlob`/`ReadBlob`) must not need to change
  when that happens.
- `StorageEngine` never calls `ProjectManager::CommitTaskOutput` itself —
  that stays the caller's job, after `WriteBlob` returns successfully.
  `StorageEngine` only ever calls `AddBlobDirectoryEntry`.
- Durability ordering is a *calling convention*, not something
  `StorageEngine` can enforce by itself — document it prominently
  everywhere it matters (class comment, `WriteBlob`'s doc comment, this
  plan) rather than trying to bake enforcement into the type system.

---

### Task 1: `StorageEngine` class skeleton + shard management

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/StorageEngine.h`
- Create: `WindowsApp.Core/SourceFiles/StorageEngine.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
class StorageEngine
{
public:
    StorageEngine();
    ~StorageEngine();
    StorageEngine(const StorageEngine&) = delete;
    StorageEngine& operator=(const StorageEngine&) = delete;

    // projectDirectory: folder containing the .vfp file.
    // projectBaseName: the .vfp file's stem (no extension) - shard files
    // are named "<projectBaseName>.NNNN.vfpdata" in that same folder.
    bool Open(const std::wstring& projectDirectory,
              const std::wstring& projectBaseName,
              ProjectManager& projectManager);
    void Close();
    bool IsOpen() const;

private:
    ProjectManager* m_projectManager = nullptr;
    std::wstring m_projectDirectory;
    std::wstring m_projectBaseName;
    int m_activeShardIndex = 1;
    HANDLE m_activeShardHandle = INVALID_HANDLE_VALUE;
    uint64_t m_activeShardOffset = 0;

    static constexpr uint64_t kMaxShardBytes = 4ull * 1024 * 1024 * 1024; // 4 GB

    std::wstring ShardPath(int shardIndex) const;
    bool OpenOrCreateActiveShard(); // finds highest existing NNNN or starts at 0001
    bool RollToNextShard();
};
```

- [ ] **Step 1: Shard filename convention**

`ShardPath(n)` returns `<projectDirectory>\<projectBaseName>.NNNN.vfpdata`
with `NNNN` zero-padded to 4 digits (`swprintf_s` or `std::format`,
matching this codebase's existing snprintf-style formatting rather than
introducing `<format>` if it's not already used elsewhere).

- [ ] **Step 2: `Open`/`OpenOrCreateActiveShard`**

On `Open`, probe shard indices starting at 1 (`ShardPath(1)`,
`ShardPath(2)`, ...) via `GetFileAttributesW` until one doesn't exist;
the active shard is the *last existing* one (or `1` if none exist yet).
Open it with `CreateFileW(..., GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ,
..., OPEN_ALWAYS, ...)`, then `GetFileSizeEx` to initialize
`m_activeShardOffset` to the current end of file (so appends resume
correctly after reopening a project - this mirrors `ProjectManager::LoadProject`
never truncating existing data).

- [ ] **Step 3: `RollToNextShard`**

Close the current handle, increment `m_activeShardIndex`, `CreateFileW`
the new shard path with `CREATE_NEW` (must not already exist - a
collision here is a programmer error, not an expected runtime condition),
reset `m_activeShardOffset` to 0.

- [ ] **Step 4: `Close`**

`CloseHandle` the active shard if open, reset all members - mirrors
`ProjectManager::CloseProject`'s reset-everything pattern.

- [ ] **Step 5: Add to `WindowsApp.Core.vcxproj`**

`ClInclude`/`ClCompile` entries for the new header/source, same pattern
as `TaskScheduler.h`/`.cpp` and `PipelineDriver.h`/`.cpp` from the prior
plan.

- [ ] **Step 6: Header/source consistency check**

Run:
```powershell
Select-String -Path WindowsApp.Core\HeaderFiles\StorageEngine.h,WindowsApp.Core\SourceFiles\StorageEngine.cpp -Pattern 'Open|Close|RollToNextShard'
```
Expected: declarations in the header, matching definitions in the source.

---

### Task 2: `WriteBlob`/`ReadBlob` (raw bytes) + `blob_directory` integration

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/StorageEngine.h`
- Modify: `WindowsApp.Core/SourceFiles/StorageEngine.cpp`

**Interfaces:**
```cpp
// Appends `length` bytes from `data` to the active shard (rolling to a
// new shard first if this write would exceed kMaxShardBytes), registers
// a blob_directory row via ProjectManager::AddBlobDirectoryEntry, and
// returns the new blobId.
//
// DURABILITY ORDERING (docs/ARCHITECTURE.md SS5, SS7.2): the caller MUST
// call ProjectManager::CommitTaskOutput(taskId, blobId) only AFTER this
// returns successfully - never before, never speculatively. If the
// process dies between this call and CommitTaskOutput, the written bytes
// are simply an orphaned, harmless blob and the owning task re-runs from
// scratch on resume (idempotency, SS7.1). The reverse ordering would mark
// a task COMPLETED with no data backing it - unrecoverable without a
// dedicated repair pass this design deliberately avoids needing.
std::optional<int64_t> WriteBlob(const void* data, size_t length, const std::string& formatTag);

// Looks up the blob's BlobDirectoryEntry, opens its shard (which may not
// be the currently-active one), and reads back the full byte range.
std::optional<std::vector<uint8_t>> ReadBlob(int64_t blobId);
```

- [ ] **Step 1: Implement `WriteBlob`**

1. If `m_activeShardOffset + length > kMaxShardBytes`, call
   `RollToNextShard()` first (Task 1).
2. `SetFilePointerEx` to `m_activeShardOffset` (defensive - the handle
   should already be positioned there from the last write/open, but an
   explicit seek makes this correct even if that invariant is ever
   broken), `WriteFile` the full buffer (loop on partial writes -
   `WriteFile` is not guaranteed to write everything in one call for
   large buffers).
3. Build a `BlobDirectoryEntry` (`shardFile` = bare filename via
   `ShardPath`'s tail, not the full path; `offset` = the pre-write
   `m_activeShardOffset`; `length`; `compressedLength` = `std::nullopt`;
   `formatTag`), call `m_projectManager->AddBlobDirectoryEntry(entry)`.
4. Advance `m_activeShardOffset += length`, return the new blob id.
5. Return `std::nullopt` on any Win32 API failure (expected-failure
   contract, matching `ITaskExecutor::Execute`'s "expected failures
   return false/empty, exceptions are for programmer errors" convention
   from the prior plan) - do not throw.

- [ ] **Step 2: Implement `ReadBlob`**

1. `m_projectManager->GetBlobDirectoryEntry(blobId)` - return
   `std::nullopt` if absent.
2. Resolve the shard's full path (`m_projectDirectory` + the entry's
   `shardFile`), open it `GENERIC_READ`/`FILE_SHARE_READ` (a separate
   handle from the active shard - the entry's shard may not be the
   active one, and even when it is, concurrent read+append must not
   fight over one file pointer).
3. `SetFilePointerEx` to the entry's `offset`, `ReadFile` `length` bytes
   into a `std::vector<uint8_t>` (loop on partial reads), close the
   handle, return the buffer.

- [ ] **Step 3: Test coverage**

In `WindowsApp.Tests/StorageEngineTests.cpp` (new file, registered in
`WindowsApp.Tests.vcxproj`):
- Round-trip: `WriteBlob` then `ReadBlob` the same bytes back, assert
  equal.
- Multiple blobs in one shard: write 3 small blobs, assert each
  `ReadBlob` returns exactly its own bytes (proves offset bookkeeping
  isn't cross-contaminating).
- `blob_directory` row exists: after `WriteBlob`, call
  `ProjectManager::GetBlobDirectoryEntry` directly and assert
  `formatTag`/`length` match what was written.
- Reopen: `Close()` the `StorageEngine`, `Open()` a fresh instance
  against the same project directory, `WriteBlob` again, assert the new
  blob's offset continues from where the previous session left off (not
  overwriting) - this is the concrete test for Task 1 Step 2's
  reopen-resumes-at-EOF behavior.

---

### Task 3: `PixelBuffer` convenience wrappers

**Files:**
- Modify: `WindowsApp.Core/HeaderFiles/StorageEngine.h`
- Modify: `WindowsApp.Core/SourceFiles/StorageEngine.cpp`

**Interfaces:**
```cpp
// Convenience wrappers for the common case of storing a decoded/rendered
// image. Framing: a fixed 8-byte header (int32 width, int32 height)
// immediately followed by the raw pixel data (matches PixelBuffer::data's
// layout) - blob_directory itself has no width/height columns (see
// docs/superpowers/plans/2026-07-07-vfp-project-schema.md's schema), so
// the dimensions travel inside the blob payload instead of widening that
// table for one consumer.
std::optional<int64_t> WritePixelBuffer(const PixelBuffer& buffer, const std::string& formatTag);
std::optional<PixelBuffer> ReadPixelBuffer(int64_t blobId);
```

- [ ] **Step 1: Implement `WritePixelBuffer`**

Build a contiguous byte buffer: `int32_t width`, `int32_t height`, then
`buffer.data`'s raw bytes (`buffer.data.size() * sizeof(unsigned short)`),
call `WriteBlob` on the whole thing.

- [ ] **Step 2: Implement `ReadPixelBuffer`**

`ReadBlob`, validate the returned buffer is at least 8 bytes, decode
`width`/`height` from the header, reinterpret the remainder as
`unsigned short` pixel data sized `(bytes.size() - 8) / sizeof(unsigned short)`,
populate and return a `PixelBuffer`. Return `std::nullopt` if the blob is
missing or too short to contain a valid header (defensive - this
indicates a format-tag mismatch or corrupt data, not a normal "not
found" case, but still an expected-failure return rather than a throw).

- [ ] **Step 3: Round-trip test**

`WritePixelBuffer` a small synthetic `PixelBuffer` (e.g. 4x3, sequential
pixel values), `ReadPixelBuffer` it back, assert `width`/`height`/`data`
all match exactly.

## Self-Review

- Spec coverage: implements the WriteFile-based half of `docs/ARCHITECTURE.md`
  §5 in full (shard rollover, `blob_directory` integration, durability
  ordering) - explicitly does not implement DirectStorage/GDeflate, which
  is called out as a scope cut, not silently dropped.
- Placeholder scan: no placeholder steps; the WriteFile path is a real,
  documented interim implementation per §5/§13's own guidance to ship
  writes conventionally first.
- Type consistency: `StorageEngine` matches `docs/ARCHITECTURE.md` §3's
  component diagram placement (`WindowsApp.Core`, alongside `ProjectManager`
  and `PipelineDriver`+`TaskScheduler`).
