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
| 4 | [2026-07-07-storage-engine.md](2026-07-07-storage-engine.md) — sharded `.vfpdata` blob container, WriteFile-based path (DirectStorage deferred) | §5 | Plan 1 |
| 5 | [2026-07-07-raw-ingest.md](2026-07-07-raw-ingest.md) — first real `ITaskExecutor`: LibRaw unpack + GPU demosaic kernel chain | §4.1 | Plan 2, Plan 4 |
| 6 | [2026-07-07-align-stage.md](2026-07-07-align-stage.md) — nvJPEG preview, FAST+BRIEF features, tensor-core RANSAC homography | §4.2 | Plan 5 |
| 7 | [2026-07-07-optimize-stage.md](2026-07-07-optimize-stage.md) — gain compensation, Reinhard color transfer, checkpointed bundle adjustment | §4.3, §7.3 | Plan 6, Plan 5 |
| 8 | [2026-07-07-render-stage.md](2026-07-07-render-stage.md) — real overlap culling, VRAM-budget chunk sizing, warp/gain/median-stack render | §4.4, §7.4 | Plan 7 |
| 9 | [2026-07-07-nvjpeg-export.md](2026-07-07-nvjpeg-export.md) — GPU JPEG preview/share export | §4.5 | Plan 6, Plan 8 |
| 10 | [2026-07-07-real-project-creation-ui.md](2026-07-07-real-project-creation-ui.md) — real multi-file RAW picker, real executors replace the demo stub | — | Plans 5-9 |

Plans 1-3 form a deliberately GPU-free foundation: by the end of Plan 3,
the app can create a project, run a (stub) pipeline in the background,
show live progress, resume after a simulated crash, and cancel
cooperatively — all the concurrency/resume/UI plumbing that's easy to get
subtly wrong — with zero CUDA or DirectStorage code written yet, and it
is implemented, built, and manually verified (issues #2-#17). Plans 4-10
replace the stub executor with real ones behind the same `ITaskExecutor`
contract; none of them change `TaskScheduler`'s or `PipelineDriver`'s
public contract or the UI thread-marshaling code from Plan 3 — Plan 10's
own header note confirms this held.

**Concrete decisions Plans 4-10 make that `docs/ARCHITECTURE.md` itself
left open** (worth knowing before reading them, so they don't read as
unexplained departures): Align's feature algorithm is FAST+BRIEF (§4.2
named it as one of two candidates, deferred the choice); a real gap was
found and fixed while writing these plans — `TaskScheduler::RegisterExecutor`
maps one executor per `PipelineStage`, so any stage needing more than one
`unit_kind` (Align, Optimize) uses a single composite executor that
dispatches internally by `unit_kind`, rather than registering multiple
executors per stage (which would silently overwrite each other); Render's
`chunk_contributors` seeding must happen after Optimize (needs final
homographies), not upfront at project creation like Ingest/Align/Optimize's
task lists — `2026-07-07-real-project-creation-ui.md` Task 1 wires this as
a small, explicit addition to `PipelineDriver::Run`'s existing stage loop;
`CudaPipeline::MultiBandBlend`'s pairwise signature doesn't fit Render's
N-way chunk contributors as-is, so Render uses `MedianStack` alone for v1
and flags the mismatch rather than force-fitting it.

**Every GPU-numeric-correctness claim in Plans 4-10 is unverified from
this environment** (no Windows, no CUDA, no GPU) — each plan says so
explicitly where it applies, and Plan 10's manual-verification step is
the first point any of it gets a real look.

## Open risks carried forward

See `docs/ARCHITECTURE.md` §13 for the full list — the two most likely to
affect plan ordering above: DirectStorage write-path maturity (may push
plan 1 above to ship read-only first) and GPU-demosaic quality parity with
LibRaw (may keep the CPU LibRaw path as the *default* for longer than §4.1
implies, gated by a config flag, until the reference-image validation
passes).
