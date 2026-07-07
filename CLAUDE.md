# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Windows desktop app for stitching large batches of RAW photos into a single
gigapixel/360 panorama. It follows the classic 3-stage panorama-stitching
pipeline (align → optimize → render), with the heavy per-pixel work offloaded
to CUDA and, where possible, tensor cores.

The repo is a native C++ MSBuild solution — there is no `.sln`/`.csproj` at the
root, only `WindowsApp.slnx` (the newer XML solution format). This is unrelated
to the .NET MES projects under `~/works/*Mes*` — do not assume ASP.NET/Blazor
conventions here.

## Build

Requires Visual Studio 2022+ with the Desktop C++ + Windows App SDK workload,
and CUDA Toolkit 13.3 (build customizations are referenced by exact version in
`WindowsApp.Compute.vcxproj`). Building the CUDA and WinUI/packaging pieces
generally requires the Visual Studio IDE or `msbuild`/`dotnet msbuild`; a
plain `dotnet build` will not work since this is not a .NET solution.

```powershell
# Build the whole solution (from repo root, in a VS Developer shell)
dotnet msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64

# Build/run just the packaged app for deployment/testing
dotnet msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64 /t:"WindowsApp (Package)"
```

Only `x64` is meaningful today — `WindowsApp.Compute` (CUDA) only has
Debug/Release **x64** configs, even though the top-level `.slnx` also lists
ARM64/x86 platforms for the other projects.

### Tests

`WindowsApp.Tests` is an MSTest native (`CppUnitTestFramework`) project — run
via Visual Studio Test Explorer or `vstest.console.exe`, not `dotnet test`.
Currently only a placeholder test (`WindowsApp.Tests/UnitTests.cpp`) exists.

## Project layout

Five MSBuild projects wired together via `WindowsApp.slnx`:

| Project | Type | Purpose |
|---|---|---|
| `WindowsApp/WindowsApp` | C++/WinRT exe | WinUI 3 desktop shell (XAML UI) |
| `WindowsApp/WindowsApp (Package)` | `.wapproj` | MSIX packaging for the exe |
| `WindowsApp.Core` | Static lib | Pipeline orchestration, RAW decoding, project DB, disk cache |
| `WindowsApp.Compute` | Dynamic lib (CUDA) | GPU kernels (warp, blend, tensor-core math) |
| `WindowsApp.Tests` | MSTest native DLL | Unit tests; references Core + Compute |

**Important: the UI is not yet wired to the engine.** `WindowsApp.vcxproj` has
no `ProjectReference` to `WindowsApp.Core` or `WindowsApp.Compute` — only
`WindowsApp.Tests` currently links against them. `MainWindow.xaml.cpp` is a
standalone image-editor-style demo (open/preview/slider stub) built per
`docs/superpowers/plans/2026-07-06-image-editor-screen.md`; it does not call
into `ProjectManager`/`WorkflowController`. When asked to connect UI actions to
the real pipeline, this link needs to be added deliberately, not assumed to
already exist.

### WindowsApp.Core — orchestration layer

- `Types.h` — shared POD types: `PixelBuffer` (16-bit RGB48/grayscale),
  `Homography` (3x3 row-major), `ChunkModel`/`ChunkStatus` (tiled render grid).
- `ImageLoader` — wraps vendored **LibRaw** (`WindowsApp.Core/libraw/`) to open
  RAW files (RAF/CR2/NEF/ARW/DNG), read EXIF metadata, and decode full-res,
  thumbnail, or ROI crops. Pimpl'd (`struct Impl`) to keep LibRaw out of the
  public header.
- `ProjectManager` — persists project state (input images + homographies,
  chunk grid + status) to a **SQLite** file (vendored `sqlite3/` amalgamation)
  so a stitching job can be resumed/inspected between runs.
- `CacheManager` — reads/writes decoded `PixelBuffer` chunks to disk (backing
  store for `ChunkModel::cache_path`) so Stage 3 doesn't have to hold the full
  output canvas in memory.
- `WorkflowController` — drives the 3-stage pipeline on a `std::jthread`
  (`std::stop_token`-cancellable), reporting progress/log/stage via
  `std::function` callbacks:
  - **Stage 1 (Align)**: low-res decode → feature extract → match → RANSAC
    homography per input image.
  - **Stage 2 (Optimize)**: gain compensation, color transfer, bundle
    adjustment (Levenberg-Marquardt, normal equations solved on GPU).
  - **Stage 3 (Render)**: per-`ChunkModel` tile — warp each contributing input
    into the tile, median-stack overlaps, multi-band (Laplacian pyramid) blend
    seams, write to cache.
  - Forward-declares `WindowsApp::Compute::CudaPipeline` and falls back to
    `m_gpuAvailable = false` when no compatible GPU is found — CPU paths for
    these stages are still expected to exist/be added.

### WindowsApp.Compute — CUDA kernels

`CudaPipeline` is the public (`COMPUTE_API` export) façade; `Kernels::*` in
`tensor_ops.cuh`/`median_stack.cuh` are the actual `__global__` kernels it
launches. Two kernel families:

- **Plain kernels** (`median_stack.cuh`): perspective warp (backward mapping +
  inverse homography), sigma-clipped median stack, separable Gaussian blur +
  Laplacian pyramid build/blend/reconstruct for multi-band blending, gain
  application, 2x up/downsample.
- **Tensor-core kernels** (`tensor_ops.cuh`, WMMA 16x16x16 tiles, requires
  SM 7.0+/Volta or newer): batched FP16 matmul, homography estimation via DLT
  (build AᵀA then solve by Cholesky), and normal-equations assembly
  (JᵀJ, Jᵀr) for the Stage 2 bundle-adjustment solve. `GpuInfo::hasTensorCores`
  gates whether these paths are used.
- Code generation is currently pinned to `compute_89,sm_89` (Ada
  Lovelace/RTX 40-series) in `WindowsApp.Compute.vcxproj` — despite
  `CudaPipeline::Initialize()` doing runtime SM-version detection, only sm_89
  is actually compiled into the binary today. Widening `CodeGeneration` is
  needed before older/newer GPUs get a native (non-JIT) code path.

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
