# v2 Architecture — Implementation Roadmap

Index of implementation plans for the ground-up redesign described in
`docs/ARCHITECTURE.md`. Each plan is independently executable
(`superpowers:subagent-driven-development` / `executing-plans`) and lists
its own dependencies; this file is the ordering and status index, not a
plan itself — it has no task checkboxes of its own.

## Plans written so far

| Order | Plan | Architecture ref | Depends on |
|---|---|---|---|
| 1 | [2026-07-07-vfp-project-schema.md](2026-07-07-vfp-project-schema.md) — `.vfp` schema, `Task`/`TaskStatus`/`PipelineStage` types, retires `WorkflowController`/`CacheManager` | §7.1, §9 | none |
| 2 | [2026-07-07-task-scheduler-core.md](2026-07-07-task-scheduler-core.md) — generic `ITaskExecutor`/`TaskScheduler`/`PipelineDriver`, proven against a stub executor | §7, §14.1 | Plan 1 |
| 3 | [2026-07-07-ui-progress-nonblocking.md](2026-07-07-ui-progress-nonblocking.md) — non-blocking UI run with live progress + cancel, first `WindowsApp` → `WindowsApp.Core` link | §14.2 | Plan 2 |

These three form a deliberately GPU-free foundation: by the end of Plan 3,
the app can create a project, run a (stub) pipeline in the background,
show live progress, resume after a simulated crash, and cancel
cooperatively — all the concurrency/resume/UI plumbing that's easy to get
subtly wrong — with zero CUDA or DirectStorage code written yet. Every
plan after this one is about replacing stub executors with real ones
behind the same `ITaskExecutor` contract; none of them should need to
change `TaskScheduler`, `PipelineDriver`, or the UI thread-marshaling code.

## Plans not yet written (next, roughly in dependency order)

Each of these should become its own `docs/superpowers/plans/YYYY-MM-DD-*.md`
file when picked up — listed here at roadmap grain only, since committing
to task-by-task detail now (before, e.g., the exact demosaic kernel
algorithm or exact DirectStorage call sequence is settled) would likely
need revision before execution anyway.

1. **StorageEngine — DirectStorage container.** `docs/ARCHITECTURE.md` §5:
   `.vfpdata` shard read/write, `blob_directory` population, CUDA↔D3D12
   interop (`cudaImportExternalMemory`/`cudaExternalSemaphore`), the
   write-before-commit durability ordering that `CommitTaskOutput` (Plan 1)
   assumes its caller upholds. Should include the write-path fallback
   (plain overlapped `WriteFile`) called out in §5 and §13 as an open risk,
   with DirectStorage writes as a follow-up once verified stable.
2. **RawIngest — LibRaw unpack + GPU demosaic.** §4.1: the first real
   `ITaskExecutor` (`unit_kind = 'image'`), replacing v1's
   `ImageLoader::DecodeROI` full-frame-per-chunk approach with a one-time
   GPU demosaic kernel chain. Needs the reference-image validation against
   LibRaw's CPU output flagged in §13 before being trusted as the sole path
   for Bayer sensors.
3. **Align (Stage 1).** §4.2: nvJPEG preview decode, feature
   extract/match, tensor-core RANSAC homography — two `ITaskExecutor`
   kinds (`'image'` for features, `'pair'` for match+solve).
4. **Optimize (Stage 2).** §4.3, §7.3: gain/color-transfer as ordinary
   per-image tasks; bundle adjustment as the one checkpointed (not
   task-parallel) executor, exercising `checkpoint_json` for the first
   time.
5. **Render (Stage 3).** §4.4, §7.4: overlap-culling populates
   `chunk_contributors` (Plan 1's table) for real instead of a test seed;
   CUDA-graph-per-chunk-shape-class; VRAM-budget-driven chunk sizing (§6).
6. **nvJPEG export path.** §4.5: GPU-side preview/share JPEG encode of
   finished output — independent of the render path's archival format, can
   land any time after Plan 5.
7. **Real project-creation UI.** Replaces Plan 3's scratch-project
   stand-in with actual multi-file RAW picker → `input_images` rows,
   surfacing real `PipelineStage`/`Task` progress instead of stub tasks.

## Open risks carried forward

See `docs/ARCHITECTURE.md` §13 for the full list — the two most likely to
affect plan ordering above: DirectStorage write-path maturity (may push
plan 1 above to ship read-only first) and GPU-demosaic quality parity with
LibRaw (may keep the CPU LibRaw path as the *default* for longer than §4.1
implies, gated by a config flag, until the reference-image validation
passes).
