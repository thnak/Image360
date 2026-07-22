# WinUI feature management (burst modes in the desktop app)

## Goal

The WinUI desktop app (`WindowsApp.exe`) only ever exposed the panorama
stitch pipeline — `MainWindow.xaml.cpp` owns a real `PipelineDriver` wired
to `STAGE0_INGEST..STAGE3_RENDER`, but the four burst modes built this
session (MFNR, HDR+, Night Sight, Super Res Zoom — all real, tested, and
already reachable from `image360_cli`) were never wired into the desktop
UI at all. Add a **feature picker** to the app so a user can choose
Panorama Stitch or any of the four burst modes from the same window, run
it, watch progress, and export the result — the same executors/kernels
the CLI and test suite already exercise, just driven from the GUI instead
of argv.

## Depends on

Phases 1-4 (all closed) — `BurstAlignExecutor`/`BurstMergeExecutor`,
`ProjectManager::CreateBurstProject`, `Tiff16Writer` (reused for burst
export, same as the CLI). No engine changes needed — this is UI-layer
wiring only, mirroring `cli/main.cpp`'s own dispatch logic.

## Scope decisions

1. **Burst modes always run on CPU, regardless of the Compute Backend
   combo box.** Confirmed while building the CLI (issue #53): every
   burst-mode kernel (`BlockMatchAlign`, `RobustMergeAccumulate`,
   `TileFftMerge`, `StructureTensorKernelRegression`) has only a
   `NOT_SUPPORTED` stub on `CudaPipeline`/`VulkanPipeline`. Selecting a
   burst-mode Feature forces CPU (with a log line explaining why, same as
   the CLI's warning) independent of whatever the Compute Backend combo
   says — Stitch keeps its existing Auto/CUDA/Vulkan/CPU behavior
   unchanged.
2. **No live chunk-by-chunk preview for burst modes.** Stitch's
   `WriteableBitmap` preview canvas is pre-sized from a coarse
   pre-alignment estimate (`totalWidth`/`totalHeight`) *before* the
   pipeline starts, because Render produces one chunk at a time into
   known sub-rectangles of that known-size canvas. A burst project has no
   chunk grid and no known output size until `BURST_MERGE` actually runs
   (Super Res Zoom upsamples; the others don't) — so burst modes show the
   empty state until `BURST_FINISH` completes, then blit the whole result
   in one shot at its real (now-known) size. Real, honest behavior for
   this pipeline shape, not a missing feature.
3. **Export supports both TIFF (lossless) and JPEG (lossy) for burst
   modes, JPEG-only for Stitch.** `PanoramaExporter::ExportPreviewJpeg`
   only ever writes JPEG (unchanged, still Stitch's only export path).
   Burst modes reuse `WriteTiff16RGB` (already built for the CLI, already
   independently verified against Pillow) plus the existing
   `IImageCodec`/`JpegCodec` narrow-and-encode path — same
   extension-driven dispatch `cli/main.cpp::WriteBurstOutput` already
   uses, not a new format decision.
4. **Renamed `Stitch*` members/handlers to feature-neutral names**
   (`RunStartButton`, `RunCancelButton`, `ExportButton`,
   `RunStatusText`, `RunProgressBar`, `m_project`, `m_storage`,
   `m_runThread`, `m_runStopSource`) — keeping `Stitch*` names once the
   same button also drives MFNR/HDR+/Night Sight/Super Res Zoom would
   mislead the next reader the same way an un-renamed `kBurstMergeSigma`
   would if it silently started controlling HDR+ too (this codebase's own
   established naming discipline). Mechanical rename, not a design
   change — behavior for Stitch itself is unchanged.
5. **File picker is unchanged for both flows.** The existing RAW +
   standard-image filter list and the per-file `CfaType` detection loop
   (`IsJpegFile`/`IsStandardImageFile`/`ImageLoader::UnpackRaw`) already
   work identically for panorama tiles and burst frames — no picker
   changes needed, same as `image360_cli` needing no format-specific
   logic either.

## Architecture

- **`MainWindow.xaml`**: new `FeatureComboBox` (Panorama Stitch / MFNR /
  HDR+ / Night Sight / Super Res Zoom) above the existing Compute Backend
  combo, wired to `FeatureComboBox_SelectionChanged`. Header title/
  subtitle and the card's title become dynamic (updated from code-behind
  on selection change) instead of the hardcoded "Panorama Stitcher"/
  "Panorama Stitch" strings. `Stitch*`-named controls renamed per scope
  decision 4.
- **`MainWindow.xaml.h`**: `AppFeature` enum (`Stitch`, `Mfnr`, `HdrPlus`,
  `NightSight`, `SuperRes`); `m_selectedFeature` member (defaults to
  `Stitch`, preserving today's default behavior for anyone who never
  touches the new combo); members renamed per scope decision 4; new
  `ShowFinalBurstResult(const PixelBuffer&)` declaration (scope decision
  2's single-shot blit).
- **`MainWindow.xaml.cpp`**:
  - `FeatureComboBox_SelectionChanged`: updates `m_selectedFeature`,
    updates the header/card title text, and — mirroring
    `ComputeBackendComboBox_SelectionChanged`'s existing
    `m_computeInitialized = false` pattern — invalidates the cached
    compute backend so the next Run re-selects with the right
    forced-CPU-or-not policy for the newly chosen feature.
  - `RunStartButton_Click` (renamed from `StitchStartButton_Click`):
    same file-picker + `PickedImage` detection loop unchanged, then
    branches on `m_selectedFeature`:
    - `Stitch`: today's exact `CreateProject`/`SeedIngestTasks`/
      `SeedAlignTasks`/`SeedOptimizeTasks` + 4-executor registration,
      unchanged.
    - Burst modes: `CreateBurstProject(path, ToBurstMode(feature))` →
      `SeedBurstAlignTasks`/`SeedBurstMergeTasks` →
      `BurstAlignExecutor` on `BURST_ALIGN` + one `BurstMergeExecutor` on
      both `BURST_MERGE`/`BURST_FINISH` (exact mirror of
      `cli/main.cpp::RunBurst`'s registration). Compute backend forced to
      `CpuComputeBackend`+`JpegCodec` (scope decision 1) instead of
      `SelectComputeBackend(preference)`.
  - `StageToDisplayString`: gains `BURST_ALIGN`/`BURST_MERGE`/
    `BURST_FINISH` cases.
  - The `onTaskCompleted` callback: existing `STAGE3_RENDER` chunk-blit
    branch unchanged; new `BURST_FINISH` branch reads the output
    `PixelBuffer` and calls `ShowFinalBurstResult` (scope decision 2)
    instead of `BlitRenderedChunk` (which assumes a pre-sized canvas and
    known chunk offset — neither holds for burst).
  - `ExportButton_Click` (renamed from `StitchExportButton_Click`):
    Stitch branch unchanged (`PanoramaExporter::ExportPreviewJpeg`,
    `.jpg`-only `FileSavePicker` choice). New burst branch: `FileSavePicker`
    offers both TIFF and JPEG choices, reads `BURST_FINISH`'s output blob,
    dispatches to `WriteTiff16RGB` or `JpegCodec::Encode`+file-write by
    the chosen extension (scope decision 3).

`ProjectManager`/`PipelineDriver`/`BurstAlignExecutor`/`BurstMergeExecutor`
need zero changes — this phase only wires existing, already-tested engine
surface into a second UI entry point.

## Tech stack

C++/WinRT, WinUI 3 XAML — same as the rest of `WindowsApp`. No new
dependency. MSBuild-only (WinUI doesn't build via the CMake path) — this
phase can't be verified via `ctest`/Linux at all, only via win-thanh's
full `WindowsApp.slnx` build and (best-effort, see Tasks) an actual
interactive run.

## Tasks

1. `MainWindow.xaml`: `FeatureComboBox` + dynamic title bindings + control
   renames.
2. `MainWindow.xaml.h`: `AppFeature` enum, renamed members,
   `ShowFinalBurstResult` declaration.
3. `MainWindow.xaml.cpp`: `FeatureComboBox_SelectionChanged`,
   `RunStartButton_Click`'s feature branch, forced-CPU backend selection
   for burst, `StageToDisplayString` burst cases, `onTaskCompleted`'s
   `BURST_FINISH` branch + `ShowFinalBurstResult`, `ExportButton_Click`'s
   burst branch.
4. Full `WindowsApp.slnx` MSBuild verification on win-thanh (0 errors) —
   this IS the build-time proof the UI/engine wiring type-checks, since
   there's no CMake path for WinUI code.
5. Best-effort interactive smoke test on win-thanh: launch
   `WindowsApp.exe`, confirm it starts and stays up (no immediate crash),
   since a full click-through isn't automatable headlessly over SSH the
   way `image360_cli`'s argv-driven smoke test was — disclosed as a real
   verification gap, not silently skipped.

## Self-review

- Does Stitch's existing behavior change at all? No — every Stitch code
  path is either untouched or a pure rename; the only new branching is on
  `m_selectedFeature != Stitch`.
- Does this duplicate `cli/main.cpp`'s burst-mode wiring logic instead of
  sharing it? Yes, by necessity — the CLI's `RunBurst` is a synchronous
  console function; the UI's version needs to interleave with
  `co_await`/dispatcher-queue callbacks and can't share a body. Both call
  the same underlying `ProjectManager`/executor/`PipelineDriver` API, so
  there's no *engine*-level duplication, only UI-glue duplication.
- Is the forced-CPU-for-burst decision surfaced to the user, not silent?
  Yes — logged via the same `AppendToStitchLog`/status-text path
  `ComputeBackendFactory`'s own fallback messages already use.
