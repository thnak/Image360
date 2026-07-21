# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A native C++ engine for turning large batches of RAW photos into a single
high-quality image. Today that means gigapixel/360 panorama stitching (the
classic 3-stage align → optimize → render pipeline). The project is actively
growing into a broader **multi-frame computational-photography engine** —
HDR+, Multi-Frame Noise Reduction, Super Res Zoom, and Night Sight, GCam-class
features built for a workstation compute budget rather than a phone SoC (so
quality is prioritized over latency). See
`docs/COMPUTATIONAL_PHOTOGRAPHY.md` for the research + design behind this
direction before starting any work in that area — it is design-only, no code
yet.

The heavy per-pixel work runs through a hardware-abstraction interface,
`IComputeBackend` (`WindowsApp.Compute/HeaderFiles/IComputeBackend.h`), with
three real implementations selected at runtime by
`ComputeBackendFactory` (CUDA → Vulkan → CPU, first available wins):
`CudaPipeline` (NVIDIA, incl. tensor cores where available), `VulkanPipeline`
(any Vulkan-capable GPU — AMD/Intel included), and a CPU backend with
runtime-dispatched AVX-512/AVX2/scalar kernels. **Every new pixel-processing
feature is expected to go through this interface and work on all three
tiers** (or explicitly document which backend(s) it's still missing — see
`docs/COMPUTATIONAL_PHOTOGRAPHY.md` §4), not assume a GPU is present.

The repo has two real build systems (see [Build](#build) below): `WindowsApp.slnx`
(MSBuild, Windows-only, the authoritative build for the packaged desktop app)
and a root `CMakeLists.txt` (additive, cross-platform, engine-only — no WinUI).
This is unrelated to the .NET MES projects under `~/works/*Mes*` — do not
assume ASP.NET/Blazor conventions here.

## Build

### MSBuild (Windows, authoritative — builds the packaged desktop app)

Requires Visual Studio 2022+ with the Desktop C++ + Windows App SDK workload,
and CUDA Toolkit 13.3 (build customizations are referenced by exact version in
`WindowsApp.Compute.vcxproj`). Building the CUDA and WinUI/packaging pieces
requires the Visual Studio IDE or the real `msbuild.exe` from a VS Developer
shell — **`dotnet msbuild` does NOT work here**, confirmed 2026-07-21: the
.NET SDK's bundled MSBuild can't resolve `$(VCTargetsPath)` for the vcxproj/
wapproj projects (`error MSB4278: ... Microsoft.Cpp.Default.props ... does
not exist`), even after `vcvars64.bat`. A plain `dotnet build` won't work
either, since this is not a .NET solution.

```powershell
# From a VS Developer Command Prompt / after vcvars64.bat (real msbuild.exe, not `dotnet msbuild`)
msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64

# Build/run just the packaged app for deployment/testing
msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64 /t:"WindowsApp (Package)"
```

Only `x64` is meaningful today — `WindowsApp.Compute` (CUDA) only has
Debug/Release **x64** configs, even though the top-level `.slnx` also lists
ARM64/x86 platforms for the other projects. See
`docs/WINDOWS_BUILD_SETUP.md` for full from-scratch setup steps.

### CMake (cross-platform, engine-only — no WinUI)

Additive to the MSBuild solution, not a replacement for it. Builds
`WindowsApp.Core` always, plus `WindowsApp.Compute`/`WindowsApp.Vulkan` when a
CUDA compiler / the Vulkan SDK + `glslc` are detected (auto-skipped
otherwise — the CPU/AVX backend always builds, so this works with **no GPU
at all**). This is what makes the engine buildable and testable headlessly
on Linux, and is the intended home for a future cross-platform CLI front end
(`docs/COMPUTATIONAL_PHOTOGRAPHY.md` §6).

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

CI (`.github/workflows/windows-engine-ci.yml`) runs this CMake build + full
`ctest` suite on a self-hosted Windows/AMD-GPU runner — CUDA auto-skipped
there, Vulkan and CPU/AVX paths both exercised for real.

### Tests

Two separate test setups — **prefer the CMake/ctest one**, it's the one
actually exercised in CI and locally verifiable without a Windows/CUDA box:

- `tests/` (CMake/ctest, cross-platform) — `engine_smoke` (SQLite +
  StorageEngine round-trip, cross-tier AVX SIMD kernel checks),
  `pipeline_e2e`/`pipeline_e2e_vulkan`/`pipeline_e2e_real_photos` (full
  4-stage pipeline against synthetic or real RAW fixtures, PSNR-gated
  quality checks), `render_blend`, `raw_probe` (manual RAW-decode
  diagnostic, not a registered test), `vulkan_smoke`. This is where new
  feature tests belong, including anything from
  `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §7.
- `WindowsApp.Tests` (MSTest native, `CppUnitTestFramework`, MSBuild-only) —
  run via Visual Studio Test Explorer or `vstest.console.exe`, not
  `dotnet test`. Currently only a placeholder test
  (`WindowsApp.Tests/UnitTests.cpp`) exists; the CMake suite above is where
  real coverage has actually been built.

## Project layout

Six MSBuild projects wired together via `WindowsApp.slnx` (plus the parallel
CMake build in `CMakeLists.txt` covering the engine subset):

| Project | Type | Purpose |
|---|---|---|
| `WindowsApp/WindowsApp` | C++/WinRT exe | WinUI 3 desktop shell (XAML UI) |
| `WindowsApp/WindowsApp (Package)` | `.wapproj` | MSIX packaging for the exe |
| `WindowsApp.Core` | Static lib | Pipeline orchestration, RAW decoding, project DB, disk cache, CPU (AVX-512/AVX2/scalar) compute backend |
| `WindowsApp.Compute` | Dynamic lib (CUDA) | NVIDIA GPU kernels (warp, blend, tensor-core math) |
| `WindowsApp.Vulkan` | Dynamic lib (Vulkan) | Cross-vendor GPU kernels (AMD/Intel/any Vulkan device) — GLSL compute shaders (`shaders/*.comp`) compiled to SPIR-V via `glslc` |
| `WindowsApp.Tests` | MSTest native DLL | Unit tests; references Core only (deliberately no Compute/Vulkan dependency) |

**The UI is wired to the engine.** `WindowsApp.vcxproj` has `ProjectReference`s
to `WindowsApp.Core`, `WindowsApp.Compute`, and `WindowsApp.Vulkan`.
`MainWindow.xaml.cpp` owns a real `PipelineDriver`, registers the four real
stage executors (`STAGE0_INGEST`..`STAGE3_RENDER`), picks a compute backend
at runtime via `ComputeBackendFactory::SelectComputeBackend()` (CUDA →
Vulkan → CPU, user-overridable via a combo box), and drives `Run()` on a
background thread with UI progress/cancel — this was a stub as of the
`docs/superpowers/plans/2026-07-06-image-editor-screen.md` plan but is not
one anymore; don't assume it's still a disconnected demo.

### WindowsApp.Core — orchestration layer

- `Types.h` — shared POD types: `PixelBuffer` (16-bit RGB48/grayscale),
  `Homography` (3x3 row-major), `ChunkModel`/`ChunkStatus` (tiled render grid).
- `ImageLoader` — wraps vendored **LibRaw** (`WindowsApp.Core/libraw/`) to open
  RAW files (RAF/CR2/NEF/ARW/DNG), read EXIF metadata, and decode full-res,
  thumbnail, or ROI crops. Pimpl'd (`struct Impl`) to keep LibRaw out of the
  public header. `JpegCodec` wraps vendored **libjpeg-turbo** for JPEG/PNG
  stitch inputs and preview encode/decode (each `Decode()`/`Encode()` call
  owns a short-lived `tjhandle` — turbojpeg handles aren't safe to share
  across threads, and this pipeline runs concurrent tasks per stage).
- `ProjectManager` — persists project state to a **SQLite** `.vfp` file
  (vendored `sqlite3/` amalgamation): input images + homographies, the chunk
  grid, and the `tasks` table below.
- `StorageEngine` — sharded blob storage (`<project>.NNNN.vfpdata` files,
  `PlatformFile.h`-backed I/O) for decoded/rendered pixel buffers — replaced
  the old per-chunk `CacheManager`/MMF design. `WriteBlob` is mutex-guarded;
  don't remove that without re-reading why (concurrent writers from
  `TaskScheduler`'s per-stage parallelism raced on it before the fix).
- `TaskScheduler` + `ITaskExecutor` + `PipelineDriver` — the generic
  resumable `Task` abstraction (`docs/ARCHITECTURE.md` §7): one row per unit
  of work, `PENDING`/`RUNNING`/`COMPLETED`/`FAILED`/`CANCELLED`, `RunStage`
  dispatches whatever isn't `COMPLETED` yet — this is the entire resume
  mechanism, including for a totally fresh project. Four real executors
  registered against `PipelineStage::STAGE0_INGEST..STAGE3_RENDER`:
  - `RawIngestExecutor` — LibRaw unpack + demosaic per input image.
  - `AlignExecutor` — feature extract/match (`FastFeatureDetector`/
    `BriefDescriptorExtractor`/`FeatureMatcher`) → RANSAC homography
    (`RansacHomography.h`, math in `HomographyMath.h`) per candidate image
    pair; "too few matches"/"RANSAC didn't converge" are valid non-failure
    outcomes for genuinely non-overlapping pairs, not stage failures.
  - `OptimizeExecutor` — gain compensation, color transfer, Levenberg-
    Marquardt bundle adjustment (`BundleAdjustment.h`/`LinearSolve.h`).
  - `RenderExecutor` — per-chunk: precomputed contributor list
    (`OverlapCulling.h`), warp each contributor (inverse homography — a
    past bug here warped with the *forward* homography, silently producing
    garbage for every non-reference image; see the render_blend/
    pipeline_e2e tests), gap-aware median-stack, multi-band (Laplacian
    pyramid, `SeamBlendKernels.h`) seam blend.
  - `HomographyMath`/`LinearSolve` intentionally run on the **host**, not
    through `IComputeBackend` — these systems are tiny enough that a
    GPU round-trip added latency without benefit (see `IComputeBackend.h`'s
    own note on this tradeoff).
- CPU compute backend (`CpuComputeBackend.h`/`CpuSimdDetect.h` +
  `BayerDemosaicKernels.h`/`MedianStackKernels.h`/`WarpPerspectiveKernels.h`/
  `GainColorOps.h`/`RgbToGray.h`) — a full `IComputeBackend` implementation
  needing no GPU at all. `CpuSimdDetect` picks AVX-512/AVX2/scalar at
  **runtime** (CPUID+XGETBV); each tier is a separate translation unit
  (MSVC has no GCC-style function multiversioning), dispatched via function
  pointers. `PanoramaExporter` writes the final stitched output.
- `TextEncoding.h`/`PlatformFile.h` — portable UTF-8/wide-string conversion
  and file I/O, centralized here so the rest of Core has no direct
  `WideCharToMultiByte`/`CreateFileW`-style Win32 calls — what makes the
  CMake/Linux build (below) possible without forking the whole layer.

### WindowsApp.Compute — CUDA kernels (one of three `IComputeBackend` impls)

`CudaPipeline : IComputeBackend` is the public (`COMPUTE_API` export)
façade; `Kernels::*` in `tensor_ops.cuh`/`median_stack.cuh` are the actual
`__global__` kernels it launches. Two kernel families:

- **Plain kernels** (`median_stack.cuh`): perspective warp (backward mapping +
  inverse homography), sigma-clipped median stack, separable Gaussian blur +
  Laplacian pyramid build/blend/reconstruct for multi-band blending, gain
  application, 2x up/downsample.
- **Tensor-core kernels** (`tensor_ops.cuh`, WMMA 16x16x16 tiles, requires
  SM 7.0+/Volta or newer): batched FP16 matmul — currently unused by any
  executor (Core's own `HomographyMath`/`LinearSolve` handle the DLT/normal-
  equations solves on the host instead, see above). `GpuInfo::hasTensorCores`
  reports availability even though nothing consumes it yet.
- Code generation is currently pinned to `compute_89,sm_89` (Ada
  Lovelace/RTX 40-series) in `WindowsApp.Compute.vcxproj` — despite
  `CudaPipeline::Initialize()` doing runtime SM-version detection, only sm_89
  is actually compiled into the binary today. Widening `CodeGeneration` is
  needed before older/newer GPUs get a native (non-JIT) code path.
- `NvJpegCodec : IImageCodec` — nvJPEG-backed preview decode/export.

### WindowsApp.Vulkan — cross-vendor GPU kernels (the second `IComputeBackend` impl)

`VulkanPipeline : IComputeBackend` is the GPU path for machines without a
CUDA-capable device — AMD/Intel GPUs, and the `win-amd` CI runner
specifically. Real GLSL compute shaders (`shaders/*.comp`, compiled to
SPIR-V by `glslc`) back the two heaviest/most embarrassingly-parallel ops,
`WarpPerspective`/`MedianStack` (plus `ApplyGain`/`DemosaicBayer`); the
remaining `IComputeBackend` methods (feature detect/describe/match, Lab
stats, Reinhard color transfer) delegate to Core's portable CPU kernels —
the same "GPU only where it clearly wins" tradeoff `CudaPipeline` makes for
`HomographyMath`/`LinearSolve`. Vulkan state (instance/device/pipelines/
buffers) is hidden behind a forward-declared `VulkanContext`, pimpl-style,
same as `CudaContext`.

### Vendored third-party code

`WindowsApp.Core/libraw/` (LibRaw, its own `.sln`/`.vcxproj`) and
`WindowsApp.Core/sqlite3/` (amalgamated `sqlite3.c`/`.h`) are vendored
in-tree, not pulled via NuGet/vcpkg. Treat them as read-only upstream code —
don't restyle or "clean up" — and check their own READMEs/licenses before
modifying.

## Working conventions

- C++20 (`stdcpp20` on VS 2022+/toolset v145, `stdcpp17` fallback on v143),
  `/bigobj`, Unicode charset throughout.
- Public Core types use `std::wstring` for paths/text (Win32-native), plain
  `std::string` for ASCII ids like `ChunkModel::id`.
- Classes exposed across the Core/Compute boundary use the pimpl idiom
  (`struct Impl`) or opaque forward-declared context structs (`CudaContext`)
  to avoid leaking CUDA or LibRaw headers into consumers.
- `docs/superpowers/plans/` holds task-by-task implementation plans written
  for the `superpowers` agentic-worker skills (subagent-driven-development /
  executing-plans) — check there for the design intent and step-by-step
  history behind recent UI/feature work before re-deriving it from the diff.
