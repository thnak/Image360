# Image360

A native C++ engine for turning batches of RAW photos into a single
high-quality image — starting with gigapixel/360° panorama stitching, and
growing into a general **multi-frame computational-photography engine**:
HDR+, Multi-Frame Noise Reduction, Super Res Zoom, and Night Sight, in the
spirit of Google Camera's computational pipeline, but built for a
workstation compute budget (multi-core AVX-512 CPU and/or a CUDA or Vulkan
GPU) instead of a phone SoC — **quality is the priority, not latency.**

## What it does today

- Loads batches of RAW files (RAF/CR2/NEF/ARW/DNG, via vendored LibRaw),
  aligns them (feature detect/match + RANSAC homography), optimizes
  exposure/color consistency and bundle-adjusts the homographies, then
  renders a tiled, seam-blended (multi-band Laplacian pyramid) panorama.
- Runs the same pipeline on whatever compute is available: an NVIDIA GPU
  (CUDA), any Vulkan-capable GPU (AMD/Intel included), or a pure CPU
  fallback (runtime-dispatched AVX-512/AVX2/scalar) — see
  [Architecture](#architecture).
- Persists project state (input images, homographies, chunk/task status) to
  a SQLite `.vfp` file so a stitching job can be cancelled and resumed.

## What's planned

See [`docs/COMPUTATIONAL_PHOTOGRAPHY.md`](docs/COMPUTATIONAL_PHOTOGRAPHY.md)
for the full research + design doc. In short: a shared dense-alignment stage
underlies four new modes, each with its own merge algorithm —

| Mode | What it does |
|---|---|
| **MFNR** | General multi-frame noise reduction — align + robustly merge a burst to denoise without changing exposure/dynamic range. |
| **HDR+** | Merge a burst of deliberately underexposed, same-exposure frames via a per-tile frequency-domain (FFT + Wiener-shrinkage) filter, then tone-map via exposure fusion — Hasinoff et al. 2016. |
| **Night Sight** | Motion-adaptive frame count/exposure metering, sharing its merge algorithm with Super Res Zoom (not HDR+'s), plus a low-light-tuned tone curve. |
| **Super Res Zoom** | Handheld multi-frame super-resolution — merges sub-pixel-shifted burst frames (natural hand tremor) onto an upsampled grid via structure-tensor kernel regression — Wronski et al. 2019. |

An optional ONNX Runtime-backed AI inference path (learned denoise/
super-resolution, e.g. NAFNet/Real-ESRGAN) is planned as a quality/speed
accelerant layered on top of the classical pipelines, not a replacement for
them. A cross-platform CLI is planned as the scriptable, headless front end
for Linux/macOS/Windows, alongside the existing WinUI 3 desktop app.

## Architecture

Five (soon six, with a CLI) components:

| Project | Type | Role |
|---|---|---|
| `WindowsApp` | C++/WinRT exe | WinUI 3 desktop shell — interactive UI |
| `WindowsApp.Core` | Static lib | Orchestration: RAW decode, project DB (SQLite), task scheduling/resume, disk cache |
| `WindowsApp.Compute` | Dynamic lib | CUDA GPU kernels (warp, blend, tensor-core math) |
| `WindowsApp.Vulkan` | Dynamic lib | Vulkan compute-shader kernels — the GPU path for non-NVIDIA hardware |
| `WindowsApp.Core`'s CPU backend | (in Core) | AVX-512/AVX2/scalar kernels, runtime-tier-dispatched, no GPU required |
| `WindowsApp.Tests` | MSTest native DLL | Unit tests |

All three compute backends implement one interface,
`WindowsApp::Compute::IComputeBackend`
(`WindowsApp.Compute/HeaderFiles/IComputeBackend.h`) — warp, median-stack,
gain, demosaic, feature detect/match, color-stat/transfer ops. A
`ComputeBackendFactory` picks CUDA → Vulkan → CPU at runtime, so the same
project builds and runs correctly regardless of what GPU (if any) is
present. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
target design of the orchestration/task/resume layer, and
[`docs/COMPUTATIONAL_PHOTOGRAPHY.md`](docs/COMPUTATIONAL_PHOTOGRAPHY.md) for
how the new burst-mode features extend this same HAL.

## Build

Two build systems, both real:

- **`WindowsApp.slnx`** (MSBuild) — the authoritative Windows build,
  produces the packaged WinUI 3 desktop app. Requires Visual Studio 2022+
  (Desktop C++ + Windows App SDK workloads) and the CUDA Toolkit. See
  [`docs/WINDOWS_BUILD_SETUP.md`](docs/WINDOWS_BUILD_SETUP.md) for full
  setup steps.

  ```powershell
  dotnet msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64
  ```

- **Root `CMakeLists.txt`** — an additive, cross-platform build of the
  engine only (`WindowsApp.Core`, plus `WindowsApp.Compute`/
  `WindowsApp.Vulkan` when CUDA/Vulkan are detected) — no WinUI, no MSIX
  packaging. This is what makes the engine buildable and testable headlessly
  on Linux/CI, and is where a future CLI front end will live.

  ```bash
  cmake -S . -B build
  cmake --build build
  ctest --test-dir build
  ```

  CUDA and Vulkan are each optional and auto-skipped if their toolchain
  isn't found (`check_language(CUDA)` / `find_package(Vulkan)` +
  `glslc`) — the CPU/AVX backend always builds, so this works with no GPU
  at all.

CI (`.github/workflows/windows-engine-ci.yml`) runs the CMake build and full
test suite on a self-hosted Windows/AMD-GPU runner — CUDA auto-skipped,
Vulkan and CPU/AVX paths both exercised.

## Testing

Every pipeline feature is expected to have both a **cross-backend numerical
parity test** (CUDA vs. Vulkan vs. AVX-512 vs. AVX2 vs. scalar, bit-exact or
within a documented tolerance) and a **synthetic-fixture, quality-gated
end-to-end test** (align→optimize→render against a known-good reference,
gated on a PSNR/SSIM threshold, not just "did it run"). See
`tests/pipeline_e2e` for the existing example — that single quantitative
check found three severe, otherwise-invisible bugs (an inverted homography,
a gap-vs-black rendering ambiguity, a white-balance channel bug) that pure
"does it crash" tests never would have. New burst-mode features follow the
same bar — see `docs/COMPUTATIONAL_PHOTOGRAPHY.md` §7.

```bash
ctest --test-dir build --output-on-failure
```

## Repo layout

```
WindowsApp/           WinUI 3 desktop shell (+ MSIX packaging project)
WindowsApp.Core/      Orchestration, RAW decode (LibRaw), CPU compute backend, SQLite project DB
WindowsApp.Compute/   CUDA GPU kernels
WindowsApp.Vulkan/    Vulkan GPU kernels (cross-vendor)
WindowsApp.Tests/     MSTest native unit tests
tests/                CMake/ctest engine tests (smoke, pipeline e2e, render blend, Vulkan)
docs/                 Architecture + build docs, task-by-task implementation plans
scripts/              Dev tooling (e.g. synthetic test-fixture generation)
```

## Docs

- [`CLAUDE.md`](CLAUDE.md) — repo conventions and orientation for AI coding agents.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — target design for the panorama-stitching engine (orchestration, task/resume model, storage).
- [`docs/COMPUTATIONAL_PHOTOGRAPHY.md`](docs/COMPUTATIONAL_PHOTOGRAPHY.md) — research + design for HDR+/MFNR/Super Res Zoom/Night Sight.
- [`docs/WINDOWS_BUILD_SETUP.md`](docs/WINDOWS_BUILD_SETUP.md) — Windows dev-machine setup.
- [`docs/superpowers/plans/`](docs/superpowers/plans/) — task-by-task implementation plans and their execution history.
