# Phase 4: CLI front end

## Goal

Per `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §6/§8: a cross-platform CLI front
end, built through the CMake path that already compiles
`WindowsApp.Core`/`WindowsApp.Compute`/`WindowsApp.Vulkan` headlessly on
Linux, exposing the panorama pipeline and all four burst modes (MFNR,
HDR+, Night Sight, Super Res Zoom) from the command line — the same split
RawTherapee uses between `rtengine` and `rawtherapee-cli`. This is the last
roadmap item with a real dependency (Phases 1-3, all done); §8 lists Phase
5 (AI façade) as depending on the same phases, not on Phase 4, so there is
no ordering constraint between them — Phase 4 chosen first because it
"becomes the natural vehicle for automated golden-image/PSNR regression
tests," directly useful for verifying everything already built.

## Depends on

Phases 1-3 + Night Sight (all closed: `BurstAlignExecutor`/
`BurstMergeExecutor` for all four `BurstMode` values; `AlignExecutor`/
`OptimizeExecutor`/`RenderExecutor`/`PanoramaExporter` for panorama). No
engine changes needed — this phase is a pure new front end.

## Scope decisions (read before the code)

1. **Output formats: JPEG (lossy, existing) + a new minimal 16-bit TIFF
   writer (lossless), not DNG.** The roadmap's §6 sketch shows
   `-o out.dng` for burst subcommands, but literal DNG encoding is a full
   RAW-mosaic container format — a substantial, separate undertaking (re-
   mosaicing already-demosaiced/merged RGB back into a Bayer-pattern DNG
   isn't just "add a `.dng` file extension case," it's a different kind of
   writer with no real justification here) and is explicitly out of scope.
   `PanoramaExporter::ExportPreviewJpeg` already only ever writes JPEG
   bytes regardless of `destPath`'s extension, so `stitch`'s output is
   JPEG-only. Burst subcommands' `BURST_FINISH` output is a 16-bit
   `PixelBuffer` with no existing writer at all — added
   `Tiff16Writer.h`/`.cpp` (new, `WindowsApp::Core`): a minimal,
   self-contained, uncompressed baseline TIFF writer (single strip, 3×16-
   bit samples/pixel, RGB) — no new third-party dependency, matches this
   project's "small, self-contained, testable" precedent for every other
   new kernel this quarter. `-o` extension selects the writer (`.jpg`/
   `.jpeg` → 8-bit narrow + `JpegCodec::Encode`; `.tif`/`.tiff` →
   `Tiff16Writer`, full 16-bit precision preserved); anything else is a
   clear CLI usage error, not a silent wrong-format write.
2. **`stitch`'s canvas size is auto-estimated, not user-specified.**
   `ProjectManager::CreateProject` requires `totalWidth`/`totalHeight`
   upfront (the panorama's world canvas isn't known until Align/Optimize
   run, but the schema needs it before Ingest starts). `WindowsApp/
   WindowsApp/MainWindow.xaml.cpp:272-273` already solves this the same
   way a real user's stitch would need to: `totalWidth = maxWidth *
   numImages`, `totalHeight = maxHeight * numImages` — a coarse, documented
   pre-alignment over-estimate (chunk grid/blob storage don't care about
   wasted canvas margin, they're derived from the input images' actual
   post-alignment placement, not from filling the estimate). The CLI
   mirrors this exact heuristic rather than requiring `--width`/`--height`
   flags or inventing a different guess.
3. **CUDA backend not wired into the CLI's CMake target yet — Vulkan/CPU
   only.** No existing CMake target links `windowsapp_compute` (it's
   always been CUDA-auto-skipped in CI and only exercised via MSBuild) —
   it's a `SHARED` library (`.dll`/`.so`), and getting a CMake-built
   executable to find it at runtime needs an explicit output-directory/
   rpath fix that has no precedent to copy in this build yet (MSBuild's
   equivalent is `WindowsApp.vcxproj`'s post-build DLL-copy target, added
   in `47aaa46`, but that's an MSBuild-specific mechanism). `windowsapp_
   vulkan` is `STATIC` (no runtime-loading question) and already has a
   proven CMake-linking precedent (`tests/pipeline_e2e_vulkan/
   CMakeLists.txt`) — the CLI's `--backend` selection mirrors that same
   `IMAGE360_HAVE_VULKAN`-gated pattern. CUDA CLI support is a tracked
   follow-up, not a silent gap — this doesn't affect the underlying
   kernels' own CUDA implementations (already real, already tested via
   MSBuild), only whether the CLI executable itself can select
   `CudaPipeline` at runtime.
4. **No project persistence / resume flag.** Each CLI invocation creates a
   project in a fresh temp directory and deletes it on exit (success or
   failure) — matching every `tests/pipeline_e2e*` scenario's own
   temp-dir lifecycle. `TaskScheduler`'s resume mechanism (§7/Architecture
   §7) is real and already tested elsewhere; exposing "resume this
   project directory across CLI invocations" is a reasonable future flag
   but not needed for a first CLI that's meant to replace one manual
   stitch/merge run, not manage long-lived projects from the shell.

## Architecture

- **`cli/main.cpp`** (new, single file — matches this project's own
  `tests/*/main.cpp` convention of one file with anonymous-namespace
  helpers, not a premature multi-file split for a first CLI).
  - Argument parsing: hand-rolled (no new third-party dependency, matching
    every other phase this quarter) — subcommand
    (`stitch`/`mfnr`/`hdrplus`/`nightsight`/`superres`), positional input
    file paths, `-o/--output <path>` (required), `--quality <int>`
    (default 90, JPEG-only subcommands/extensions), `--chunk-size <int>`
    (override; default derived from the selected backend's `GpuInfo`, same
    `RecommendedChunkSize`/`RecommendedChunkSizeForCpu` split
    `ComputeBackendFactory` already uses), `--backend auto|vulkan|cpu`
    (default `auto`).
  - CfaType detection: mirrors `MainWindow.xaml.cpp`'s existing per-file
    logic (`IsJpegFile`/`IsStandardImageFile` → `STANDARD_RGB` +
    `GetStandardImageDimensions`; else `ImageLoader::Open`+`UnpackRaw` →
    `RawPlane::cfaType`/width/height, falling back to `GetMetadata` if
    `UnpackRaw` fails) — not extracted into a shared Core helper (only two
    call sites exist; a header-only rule doesn't earn an abstraction yet).
  - Backend selection: a CLI-local function (not a reuse of `WindowsApp/
    WindowsApp/ComputeBackendFactory.cpp`, which unconditionally links
    `CudaPipeline`/`NvJpegCodec` — safe only in the always-CUDA-toolkit
    MSBuild build, wrong for a CMake build that might have neither GPU
    backend compiled in) tries `VulkanPipeline` (if
    `IMAGE360_CLI_HAVE_VULKAN` compiled in) then falls back to
    `CpuComputeBackend` (always available) — paired with the portable
    `JpegCodec` either way (`VulkanPipeline` has no GPU JPEG codec of its
    own, same precedent `ComputeBackendFactory.cpp` already sets).
  - `RunStitch(...)`: `CreateProject` (auto-estimated canvas per scope
    decision 2) → `AddInputImage` per file → `SeedIngestTasks`/
    `SeedAlignTasks`/`SeedOptimizeTasks` → register
    `RawIngestExecutor`/`AlignExecutor`/`OptimizeExecutor`/
    `RenderExecutor` → `PipelineDriver::Run` → on success,
    `PanoramaExporter::ExportPreviewJpeg` to `-o` (must end in `.jpg`/
    `.jpeg`).
  - `RunBurst(mode, ...)`: `CreateBurstProject(mode)` → `AddInputImage`
    per file → `SeedBurstAlignTasks`/`SeedBurstMergeTasks` → register
    `BurstAlignExecutor` (`BURST_ALIGN`) + one `BurstMergeExecutor`
    instance for both `BURST_MERGE`/`BURST_FINISH` → `PipelineDriver::Run`
    → on success, read `BURST_FINISH`'s output `PixelBuffer` and write it
    via the extension-selected writer (scope decision 1).
  - Progress: `PipelineDriver::Initialize`'s progress callback prints
    `[stage] NN%` to stdout; the log callback prints to stderr — same
    split every `tests/pipeline_e2e*` scenario already uses, just routed
    to real stdout/stderr instead of test output.
  - Exit codes: `0` success; `1` pipeline failure
    (`PipelineDriver::GetCurrentStage() != COMPLETED` or any setup step
    returning `false`); `2` CLI usage error (bad/missing args, unsupported
    output extension, zero input files) — checked and reported before any
    expensive work starts.
- **`WindowsApp.Core/HeaderFiles/Tiff16Writer.h`/`.cpp`** (new): `bool
  WriteTiff16RGB(const std::wstring& path, const unsigned short* data, int
  width, int height)` — minimal uncompressed baseline TIFF (little-endian
  "II" byte order; required tags 256/257/258/259/262/273/277/278/279 plus
  282/283/296 for broader-reader compatibility), single strip, no
  compression. A real, testable, bounded (~150 line) self-contained
  writer, not a build-time dependency on libtiff or similar.
- **`cli/CMakeLists.txt`** (new): `add_executable(image360_cli main.cpp)`,
  links `windowsapp_core` always + `windowsapp_vulkan` when
  `IMAGE360_HAVE_VULKAN` (with `target_compile_definitions(... PRIVATE
  IMAGE360_CLI_HAVE_VULKAN)`), added via root `CMakeLists.txt`'s
  `add_subdirectory(cli)`.

## Tech stack

C++20, no new third-party dependency (hand-rolled arg parsing, hand-rolled
TIFF writer). CMake-only for now — no MSBuild `.vcxproj` for the CLI in
this phase (the CMake build is the one this front end is explicitly meant
to live in per §6; adding it to `WindowsApp.slnx` too is a reasonable
follow-up but not required — the Windows-authoritative MSBuild build stays
focused on the packaged WinUI app).

## Tasks

1. `Tiff16Writer.h`/`.cpp` + a structural round-trip test (write a known
   small buffer, manually re-parse the header/IFD/tag values/pixel data
   from the raw bytes — no TIFF-reading library exists in-tree, so the
   test IS the reader, proving byte-for-byte correctness against the TIFF6
   spec's field layout).
2. `cli/main.cpp`: arg parsing, CfaType detection, backend selection.
3. `RunStitch` (panorama subcommand).
4. `RunBurst` (mfnr/hdrplus/nightsight/superres subcommands).
5. `cli/CMakeLists.txt` + root `CMakeLists.txt` wiring.
6. Manual smoke test: run each of the 5 subcommands against small
   synthetic/fixture inputs on Linux, confirm real output files with
   correct dimensions/non-garbage pixel data (not just "exit code 0").
7. Dual-platform verification (Linux ctest, then win-thanh: CMake+ctest,
   MSBuild — the CLI itself isn't in `WindowsApp.slnx` so MSBuild
   verification here just confirms the existing solution build is
   unaffected — WindowsApp.Tests).

## Self-review

- Does this touch any existing executor/kernel code? No — pure new
  front-end + one new, independent, testable writer utility.
- Does skipping CUDA in the CLI regress the "hardware-agnostic by
  construction" principle? No — that principle governs the kernels
  themselves (already real on all 3 tiers, or documented `NOT_SUPPORTED`
  gaps); the CLI's own backend-selection wiring is a separate, narrower,
  explicitly-tracked scope cut, not a kernel-level gap.
- Is `stitch`'s auto-estimated canvas a regression vs. requiring an
  explicit size? No — it's the exact heuristic the existing WinUI app
  already uses for the same problem, not a new guess.

## Open scope cuts

- DNG output (see scope decision 1).
- CUDA backend selection in the CLI's CMake target (see scope decision 3)
  — needs a DLL/rpath output-directory fix with no existing CMake
  precedent in this repo.
- Project persistence/resume across CLI invocations (see scope decision
  4).
- `WindowsApp.slnx`/MSBuild integration for the CLI executable itself
  (CMake-only for this phase, per Tech stack above).
