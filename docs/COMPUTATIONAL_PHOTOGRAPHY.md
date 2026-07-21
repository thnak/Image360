# Computational Photography Roadmap — HDR+, MFNR, Super Res Zoom, Night Sight

> Status: **research + design, no code yet**. Companion to `docs/ARCHITECTURE.md`
> (the panorama-stitching v2 redesign). This document is the reference for
> extending Image360 from a panorama stitcher into a general multi-frame
> computational-photography engine — GCam-class features (HDR+, Multi-Frame
> Noise Reduction, Super Res Zoom, Night Sight), but built for a workstation
> compute budget (multi-core AVX-512 CPU, and/or a CUDA or Vulkan GPU) instead
> of a phone SoC, so **quality is the priority over latency**. Research was
> gathered from the original papers, public open-source reimplementations,
> and precedent from other multi-backend imaging projects — see §10 for
> sources.

## 1. Vision & scope

- **Hardware-agnostic by construction.** Every new algorithm must run
  through the existing `IComputeBackend` HAL
  (`WindowsApp.Compute/HeaderFiles/IComputeBackend.h`) and its three
  implementations — `CudaPipeline` (NVIDIA), `VulkanPipeline` (any Vulkan
  GPU, e.g. AMD/Intel), `CpuComputeBackend` (AVX-512/AVX2/scalar,
  runtime-dispatched) — selected by `ComputeBackendFactory`'s fallback
  chain. "Use any hardware if possible" is not a slogan here, it's the same
  constraint the panorama pipeline already satisfies; new features must
  satisfy it too, not bypass it with a GPU-only shortcut.
- **Two front ends, one engine.** WinUI 3 (`WindowsApp`) stays the
  interactive desktop surface; a new cross-platform CLI (§6) becomes the
  scriptable, testable, headless front end for Linux/macOS/Windows —
  matching the split RawTherapee uses between `rtengine` and
  `rawtherapee-cli` (see §4).
- **Tests are not optional.** Every new algorithm ships with (a) a
  cross-backend numerical parity test and (b) a synthetic-fixture,
  quality-gated (PSNR/SSIM) end-to-end test, following the pattern already
  proven in `tests/pipeline_e2e` (see the project memory: that single
  quality gate found 3 severe, otherwise-invisible pipeline bugs). A
  feature without both is not done.
- **AI-assisted, not AI-only.** A learned-inference path (§5) is an
  optional accelerant/quality booster layered on top of a solid classical
  pipeline, not a replacement for it — consistent with "quality is
  priority" over chasing the smallest/fastest model.

## 2. Feature research summary

### 2.1 HDR+ (burst, single-exposure merge)

Google's HDR+ (Hasinoff et al., *Burst photography for high dynamic range
and low-light imaging on mobile cameras*, SIGGRAPH Asia 2016 —
[paper PDF](https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus.pdf),
[supplement](https://static.googleusercontent.com/media/hdrplusdata.org/en//hdrplus_supp.pdf))
captures a burst of **deliberately underexposed, same-exposure** raw frames
and merges them, rather than fusing frames shot at different EVs. Three
stages, precisely (corrected from an earlier draft of this doc that
oversimplified merge as a weighted average — it is not):

- **Align**: tile-based patch search on a 4-level Gaussian pyramid (16×16
  tiles, 8×8 at the coarsest level), on 2×2-averaged downsampled grayscale.
  Finest level is brute-force L1 SAD, ±1px, pixel-precision — the paper
  explicitly tested and *rejected* phase correlation here ("outperformed by
  a well-implemented brute-force procedure"). Coarser levels use L2 SSD
  accelerated by a small per-tile FFT/convolution-theorem trick (not phase
  correlation), ±4px, with a closed-form 3×3 quadratic fit for sub-pixel
  refinement. Entirely per-tile, embarrassingly parallel — fits the existing
  kernel-per-tile style directly.
- **Merge**: operates per-tile (16×16, 32×32 in dark scenes, 50%-overlapped,
  raised-cosine windowed) on **independent Bayer color planes, pre-demosaic**.
  This is **not** a weighted temporal average — it's a per-frequency-bin
  Wiener-shrinkage filter: each tile gets a 2D DFT, and each alternate
  frame's contribution is blended against the reference tile via
  `A_z(ω) = |D_z(ω)|²/(|D_z(ω)|² + cσ²)`, with σ² from a calibrated
  Poisson+Gaussian per-ISO noise model. The paper is explicit this must be
  per-frequency-bin, not a scalar per-tile weight, so one misaligned frame
  can't poison a whole tile. **This is a genuinely new kernel family** — a
  batched small 2D real FFT/iFFT plus a complex-domain shrinkage op — not
  expressible as a variant of the existing `MedianStack`.
- **Finish**: a 13-step ordered pipeline (black-level → lens-shading →
  white-balance → demosaic *after* merge → bilateral chroma denoise → color
  correction → tone mapping → dehaze → global contrast → chromatic-
  aberration fix → unsharp-mask sharpen → hue adjustment → dithering). Tone
  mapping specifically is **exposure fusion** (Mertens et al. 2007) —
  Laplacian-pyramid blend of two synthetic exposures derived from the merged
  image, explicitly chosen over local Laplacian filters as cheaper. This
  step **does** map directly onto the multi-band blend machinery this repo
  already has (`median_stack.cuh` / `WindowsApp.Vulkan/shaders/`).

This is a genuinely different problem from classic **exposure-bracket HDR
fusion** (Debevec/Kalantari-style, multiple different-EV frames merged —
what `HDR-Transformer-PyTorch` and `Deep-HDR-with-Pytorch`, both linked
below, implement). Both are worth having eventually: HDR+ as the primary
"burst" mode (denoise + dynamic-range recovery from one exposure setting,
robust to handheld motion), and a secondary bracket-fusion mode for users
who prefer to shoot AEB brackets. They should not be conflated into one
pipeline stage.

**OSS references (ranked by how directly useful they are here):**
- [timothybrooks/hdr-plus](https://github.com/timothybrooks/hdr-plus) (MIT,
  **C++/Halide**) — a full align/merge/finish CPU implementation, the
  closest-language reference of anything found; targets a DSLR-burst
  variant of the paper. Worth reading before writing the FFT-merge kernel,
  even though its Halide schedules won't port directly.
- [hdrplusdata.org](https://hdrplusdata.org/) — Hasinoff's own writeup and
  the official [dataset](https://hdrplusdata.org/dataset.html) (3,640
  bursts / 28,461 DNGs, **CC BY-SA 4.0**) — the ground-truth source to
  validate any reimplementation's output against (§7.3).
- [martin-marek/hdr-plus-pytorch](https://github.com/martin-marek/hdr-plus-pytorch)
  (MIT) — simplified align+merge only (no finish stage), GPU via PyTorch
  tensor ops; ~200ms/frame aligning a 20MP raw on a free Colab GPU, useful
  as a rough perf baseline. Its author also ships a commercial macOS app
  ("Burst Photo") with a fuller pipeline — a real-world existence proof
  this is viable to ship, not just a research demo.
- [martin-marek/hdr-plus-swift](https://github.com/martin-marek/hdr-plus-swift)
  (**GPL-3.0**) — full pipeline, Metal GPU. Algorithm reference only, same
  license caveat as `ImageStackAlignator` in §2.4 — this repo's own license
  is unset (§9), so don't copy GPL code.
- [amonod/hdrplus-python](https://github.com/amonod/hdrplus-python)
  (**AGPL-3.0**) — companion to an IPOL 2021 paper, self-described
  "simplified" finish stage. Same license caveat.
- Exposure-bracket alternative for later: **HDR-Transformer**
  ([liuzhen03/HDR-Transformer-PyTorch](https://github.com/liuzhen03/HDR-Transformer-PyTorch),
  ECCV 2022) — dual-branch (windowed-Transformer global branch +
  channel-attention local branch) ghost-free fusion of differently-exposed
  frames; license not stated in the repo, needs clarifying before any
  reuse. **Deep-HDR-with-Pytorch**
  ([CharlieMarcotte/Deep-HDR-with-Pytorch](https://github.com/CharlieMarcotte/Deep-HDR-with-Pytorch))
  reproduces Kalantari & Ramamoorthi 2017 (optical-flow alignment + a small
  CNN fusion net); no license stated either.
- Learned burst denoise, if the AI path (§5) ever grows a dedicated
  denoiser: [google/burst-denoising](https://github.com/google/burst-denoising)
  (KPN, **Apache-2.0**, archived/TF1 — license is clean but the code is
  old), [akshaydudhane16/Burstormer](https://github.com/akshaydudhane16/Burstormer)
  (**MIT**). `goutamgmb/deep-burst-sr`/`deep-rep` are **CC-BY-NC-SA** —
  same non-commercial caveat as Restormer in §5, reference only.

**Kernel fit against `IComputeBackend`:** alignment is dense per-tile block
matching (new op — distinct from the panorama path's sparse
`DetectAndDescribeFeatures`/`MatchFeatures`/RANSAC-homography, which
recovers one global transform, not a per-tile local one; shareable with
MFNR's alignment need, §2.3). **Merge needs a genuinely new kernel
primitive class this repo doesn't have yet: batched small-tile 2D real
FFT/iFFT plus complex-domain Wiener shrinkage** — this is the single
biggest new piece of infrastructure HDR+ requires, bigger than any per-pixel
kernel extension. (A spatial-domain approximate merge is possible as a
lower-fidelity fallback, but it diverges from the documented algorithm and
should be labeled as such, not presented as equivalent.) Exposure-fusion
tone mapping in Finish reuses the existing Laplacian-pyramid multi-band
blend machinery directly.

### 2.2 Night Sight

Night Sight ([Google AI blog](https://research.google/blog/night-sight-seeing-in-the-dark-on-pixel-phones/))
is **not simply HDR+ with different parameters** — it shares HDR+'s
foundations but adds real new pieces:

- **Motion metering**: a pre-capture stage running optical flow on
  viewfinder frames to pick burst length (6 frames on tripod up to 15
  handheld) and per-frame exposure (up to 333ms handheld, 1s on tripod,
  ~6s total budget) adaptively — this is orchestration logic (in
  `PipelineDriver`/a new executor), reusing the existing feature-match
  machinery for the flow estimate, not a new GPU kernel family.
- **Merge algorithm swap**: on Pixel 3 and later, Night Sight replaces
  HDR+'s FFT/Wiener-shrinkage merge (§2.1) with the **Super Res Zoom**
  algorithm (§2.4) — structure-tensor-oriented anisotropic kernel
  regression that reconstructs full-resolution RGB directly from raw Bayer
  data, standing in for demosaic entirely rather than following it. On
  tripod, OIS is deliberately jittered to manufacture the sub-pixel shifts
  this needs. **This means Night Sight's real dependency is Super Res
  Zoom's kernel-regression merge, not HDR+'s FFT merge** — sequence
  accordingly in the roadmap (§8).
- **Tone mapping**: a "painterly" S-curve (crushed shadows, darkened
  surrounds) plus a learned auto-white-balance classifier, distinct from
  HDR+'s exposure-fusion curve.
- Astrophotography mode is a further extension of the same pipeline (up to
  16s/frame, 15 frames, ≤4min total —
  [blog](https://research.google/blog/astrophotography-with-night-sight-on-pixel-phones/)),
  not a separate design concern for this roadmap.

Implement as: motion-metering executor (new, orchestration-level) + Super
Res Zoom's merge core (§2.4, built once) + a distinct tone-mapping variant —
not as a parameter set on top of HDR+'s FFT-merge pipeline.

### 2.3 Multi-Frame Noise Reduction (MFNR) — the shared substrate

**Correction from an earlier draft of this doc:** MFNR is *not* a clean
shared substrate for all three other modes — HDR+'s merge (FFT/Wiener
shrinkage, §2.1) and Night-Sight/Super-Res-Zoom's merge (structure-tensor
kernel regression, §2.2/§2.4) are two genuinely different algorithms, not
parameter variants of one merge kernel. What *is* genuinely shared across
all four modes is the **alignment stage** (dense per-tile block matching,
§2.1's pyramid search) — build that once. For MFNR's own merge (no HDR/SR
need, just denoise a normally-exposed burst), the simplest correct choice is
a **robust weighted merge** — closer to a generalized `MedianStack` than
either of the other two algorithms — making MFNR's merge the cheapest of
the three to implement and the right one to build first (§8 Phase 1),
*not* because HDR+/Night Sight reuse it, but because it's the smallest
correct slice through the new alignment infrastructure. HDR+'s FFT merge
and Super-Res-Zoom's kernel-regression merge are each their own follow-up
phase (§8 Phases 2-3), sharing only the alignment stage with Phase 1, not
the merge kernel.

General alignment tradeoff worth deciding up front: **block/patch-based
alignment** (what HDR+ and Wronski's Super Res Zoom both use) is
noise-resilient and cheap but only models local translation; **optical-flow**
alignment handles true non-rigid subject motion better in principle but
degrades badly under noise/large motion (halos/ghosting at moving-object
boundaries). Both Google papers deliberately choose block matching plus a
*separate* statistical robustness/outlier-rejection stage over trying to
make alignment itself track arbitrary motion — same choice this project
should make, reusing the panorama path's existing global
feature-match/RANSAC for camera-shake compensation and adding a new local
block-match op only for the sub-tile residual motion robust-merge needs.

### 2.4 Super Res Zoom (handheld multi-frame super-resolution)

Google's *Handheld Multi-Frame Super-Resolution* (Wronski et al., SIGGRAPH
Asia 2019 — [arXiv:1905.03277](https://ar5iv.labs.arxiv.org/html/1905.03277),
[project page](https://sites.google.com/view/handheld-super-res/)). Natural
hand tremor gives each burst frame a slightly different sub-pixel sample
position relative to the Bayer grid, effectively oversampling the scene.
Pipeline stages: (a) pyramid block-matching alignment refined by a few
Lucas-Kanade iterations for sub-pixel accuracy (sequential/global, must run
first); (b) local gradient structure-tensor analysis (per-pixel,
parallel); (c) anisotropic kernel construction from the structure tensor,
elongated along edges (per-pixel, parallel); (d) kernel-regression
accumulation of every frame's contribution onto an upsampled output grid
(per-pixel/per-tile, order-independent — parallel); (e) the same kind of
robustness/outlier weighting as HDR+/MFNR (per-pixel, parallel, depends on a
precomputed noise model). Only stage (a) is inherently sequential/global;
everything downstream matches the existing GPU-kernel-per-pixel-or-tile
style this project already uses.

**OSS references:**
- [Jamy-L/Handheld-Multi-Frame-Super-Resolution](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution)
  (MIT) — faithful non-official reimplementation (Python + Numba CUDA),
  explicitly written for algorithmic clarity rather than production
  performance; a good line-by-line reference for the kernel math.
- [kunzmi/ImageStackAlignator](https://github.com/kunzmi/ImageStackAlignator)
  (**GPL-3.0**) — a full C#+CUDA RAW→TIFF pipeline implementing the same
  paper. **License flag:** GPL-3.0 means its *code* cannot be copied into
  this project without the whole result becoming GPL (this repo's own
  license is currently unset — see §9 open questions). Treat it as an
  algorithm reference only, implement independently.
- Deep-learning alternative family (different technique, worth tracking as
  a possible future AI-backend mode, §5): NTIRE burst-SR challenge repos —
  [goutamgmb/deep-burst-sr](https://github.com/goutamgmb/deep-burst-sr),
  [BSRT](https://github.com/Algolzw/BSRT) (flow-guided deformable
  alignment + attention fusion), trained on the public BurstSR dataset
  ([arXiv:2101.10997](https://arxiv.org/abs/2101.10997)).

**Kernel fit:** the block-matching alignment stage is genuinely new (dense
local sub-pixel translation field, not a single global homography — a
different op from both the panorama path's RANSAC homography and the MFNR
block-match, though closely related and possibly shareable). Structure
tensor, anisotropic kernel construction, kernel-regression accumulation, and
robustness weighting are all new per-pixel/per-tile kernels, but
embarrassingly parallel and a good fit for the existing
per-backend-kernel-authoring model (§3).

## 3. Shared substrate: a new "Burst Merge" pipeline family

Reuse the resumable `Task`/`TaskScheduler`/`.vfp` machinery
`docs/ARCHITECTURE.md` §7 designed (and the shipped CPU/Vulkan/CUDA
backends) for a second pipeline family alongside panorama stitching, rather
than inventing new orchestration:

- New `PipelineStage` values: `BURST_ALIGN`, `BURST_MERGE`, `BURST_FINISH`
  (tone map / sharpen), parallel to the existing `INGEST/ALIGN/OPTIMIZE/
  RENDER`. A project's *type* (panorama vs. burst-mode, and which burst mode
  — MFNR/HDR+/NightSight/SuperRes) is new `.vfp` metadata, not a new file
  format. `BURST_MERGE`'s executor is selected by mode, since the merge
  algorithm itself differs per §2.3's correction — not a single shared
  kernel with different parameters.
- New `IComputeBackend` ops needed (naming TBD at implementation time),
  grouped by which modes actually share them (§2.3):
  - `BlockMatchAlign` (dense local per-tile translation field, pyramid
    search) — shared by **all four modes** (MFNR/HDR+/Night Sight/Super Res
    Zoom); build once in Phase 1.
  - `RobustMergeAccumulate` (noise-weighted accumulation, generalizes
    `MedianStack`) — **MFNR only** (§8 Phase 1).
  - `TileFftMerge` (batched small 2D real FFT/iFFT + complex-domain Wiener
    shrinkage) — **HDR+ only** (§8 Phase 2), a genuinely new primitive
    class, not an extension of an existing op.
  - `StructureTensorKernelRegression` (structure-tensor eigendecomposition +
    anisotropic kernel construction + kernel-regression accumulation onto
    an upsampled grid) — **Super Res Zoom and Night Sight** (§8 Phase 3);
    Night Sight adds motion-metering orchestration and a distinct tone
    curve on top of this same merge kernel, it does not get its own merge
    op.
  - `LocalToneMap` (exposure-fusion variant for HDR+, painterly-curve
    variant for Night Sight) — reuses existing Laplacian-pyramid multi-band
    blend machinery, not a new kernel family.
  
  Each new op needs CUDA + Vulkan + AVX implementations to keep hardware
  parity — see §4 for how to sequence that without silently shipping uneven
  backend coverage.

## 4. HAL scaling strategy

Research validated the current `IComputeBackend` design (one pure-virtual
interface, `ComputeBackendFactory` CUDA→Vulkan→CPU fallback) against
established multi-backend imaging projects:

- **darktable** is the closest real-world analog — each pixelpipe module
  optionally implements a GPU (`process_cl`) path alongside a mandatory CPU
  path, with automatic fallback on failure. At ~100-module scale, this
  pattern works but darktable's own history shows the failure mode to watch
  for: some modules' GPU paths went stale/abandoned and now silently run
  CPU-only. **Mitigation to adopt now, before it happens here:** track a
  backend-coverage matrix per op explicitly (a table in this doc or a
  checked-in status file), and make new ops CPU-first + one GPU backend
  intentionally, flagging the missing backend as a tracked gap rather than
  an invisible one.
- **Halide** and **oneAPI/SYCL** solve the same problem by compiling one
  algorithm description to every backend (schedule/codegen separation)
  instead of hand-writing each backend — a fundamentally different, heavier
  investment. Not worth adopting now (this project's op count is still
  small and hand-written CUDA/GLSL/AVX kernels are working fine), but worth
  revisiting if kernel-authoring burden becomes the actual bottleneck once
  §2's new ops are all implemented.
- **ONNX Runtime's Execution Provider model** (ordered EP list, per-node
  graph partitioning, CPU EP as universal fallback) is architecturally
  similar to `ComputeBackendFactory` but operates at a different
  granularity (per-graph-node vs. per-whole-op) — informs §5's design
  directly rather than this section's hand-written-kernel HAL.
- **FFmpeg's `AVHWDeviceContext`/`AVHWAccel`** (CUDA/VAAPI/DXVA/VideoToolbox/
  Vulkan) and **RawTherapee's `rtengine` core + separate GUI/CLI
  executables** are both direct structural precedents for, respectively,
  this project's compute HAL and its planned Core/CLI/WinUI split (§6) —
  worth a closer read if either area needs a redesign later.

## 5. AI/ML inference integration

Treat inference as its **own façade, parallel to `IComputeBackend`, not
merged into it** — this mirrors both how ONNX Runtime already owns its own
backend selection, and how darktable 5.6 (2026) added exactly this kind of
subsystem in practice:

- **darktable 5.6** shipped an optional (`-DUSE_AI=ON`), `dlopen`-loaded-at-
  runtime ONNX Runtime subsystem (CUDA/ROCm-MIGraphX/OpenVINO/CPU EPs,
  configurable) for local denoise/upscale/masking — no hard dependency, no
  crash on missing/incompatible drivers, models distributed separately from
  the app binary via [darktable-org/darktable-ai](https://github.com/darktable-org/darktable-ai).
  Adopt the same shape here: optional build flag, runtime EP selection, and
  **do not bundle model weights in the installer** — a versioned,
  separately-downloaded asset bundle instead.
- **EP priority chain** (mirrors `ComputeBackendFactory`'s CUDA→Vulkan→CPU
  philosophy, but as ONNX Runtime's own mechanism): TensorRT EP or CUDA EP
  first on NVIDIA, DirectML/Windows ML EP on Windows for any DX12 GPU
  (AMD/Intel included — DirectML is in "sustained engineering" with new
  development moving to Windows ML, worth tracking), CPU EP as the
  universal fallback.
- **Model choices:** [NAFNet](https://github.com/megvii-research/NAFNet)
  (MIT, pretrained on SIDD/GoPro) as the default learned denoiser — cleanly
  licensed for shipping. **Avoid Restormer** for anything shipped — its
  weights are CC-BY-NC-SA, non-commercial only. For learned super-resolution,
  [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (permissive
  license, ONNX exports available, e.g. on Hugging Face) as the default,
  with lighter distilled variants as a lower-VRAM/CPU-tier fallback.
- **Positioning:** the AI path is an optional quality/speed accelerant
  layered on the classical pipelines in §2–§4 (e.g. a learned-denoise
  pre-pass, or a learned-SR blend option in the Super Res Zoom finish
  stage), never the only implementation of a feature — keeps the
  "any hardware, quality first" classical path as the guaranteed baseline.

## 6. CLI design

A new cross-platform CLI front end, built through the CMake path that
already compiles `WindowsApp.Core`/`WindowsApp.Compute`/`WindowsApp.Vulkan`
headlessly on Linux (`CMakeLists.txt` at repo root) — a thin
argument-parsing shell over `PipelineDriver`/`ProjectManager`, the same
split RawTherapee uses between `rtengine` and `rawtherapee-cli` (§4).

Sketch:

```
image360-cli stitch     <raw-files...> -o out.tif        # existing panorama pipeline
image360-cli mfnr       <burst-raw-files...> -o out.dng  # §2.3
image360-cli hdrplus    <burst-raw-files...> -o out.dng  # §2.1
image360-cli nightsight <burst-raw-files...> -o out.dng  # §2.2
image360-cli superres   <burst-raw-files...> --scale 2 -o out.dng  # §2.4
```

Each subcommand drives the same resumable `Task`/`TaskScheduler` pipeline
headlessly, printing stage/progress to stdout and using exit codes for
scripting/CI — and becomes the natural vehicle for automated golden-image/
PSNR regression tests (§7), extending the existing `tests/pipeline_e2e_*`
pattern without needing WinUI at all.

## 7. Testing strategy

Every new algorithm added under this roadmap needs, before being called
done:

1. **Cross-backend parity test** — same pattern as
   `tests/engine_smoke`'s `RunCrossTierSimdKernelChecks` (bit-exact/near-
   exact cross-check across AVX-512/AVX2/scalar), extended to also cross-
   check CUDA vs. Vulkan vs. CPU for each new op, with an explicit,
   documented numerical tolerance for float-heavy kernels (structure
   tensor, kernel regression) where bit-exactness isn't realistic across
   different GPU vendors' floating-point behavior — decide the tolerance
   policy up front, don't discover it ad hoc per test.
2. **Synthetic-fixture, quality-gated end-to-end test** — same pattern as
   `tests/pipeline_e2e`'s PSNR-vs-reference scenario, which found 3 severe,
   otherwise invisible bugs (inverted homography, gap-vs-black combine
   ambiguity, a white-balance channel zeroing bug) purely because it
   inspected rendered pixel content instead of just "did it run." A new
   burst-mode feature needs the equivalent: a synthetic burst with a known
   ground truth, PSNR/SSIM threshold set from the actual achieved value on
   a correct implementation (not guessed).
3. **Reference validation against public datasets**, license permitting —
   e.g. [hdrplusdata.org](https://hdrplusdata.org/)'s public burst set for
   HDR+/Night Sight, the BurstSR dataset for Super Res Zoom — as a periodic
   manual check, not necessarily a CI gate (real photographic datasets are
   large and slow, unlike the small synthetic DNG fixtures `tests/
   pipeline_e2e` uses).

## 8. Phased roadmap

| Phase | Scope | Depends on |
|---|---|---|
| 0 | `.vfp` schema additions (burst project type, `BURST_*` stages) + generalize `TaskScheduler`/`PipelineDriver` for a second pipeline family. No new kernels — deliberately GPU-free, mirrors how the v2 panorama redesign's Plans 1-3 proved the concurrency/resume plumbing before any CUDA code. | none |
| 1 | **`BlockMatchAlign`** (§3, shared by all four modes) + **MFNR** (§2.3) via `RobustMergeAccumulate`. Implement CPU/AVX first (fastest to correctness-test), then whichever GPU backend matches available dev hardware; track the other explicitly as a known gap (§4), don't let it go silent. Ship the quality-gated e2e test (§7) before calling this phase done. | Phase 0 |
| 2 | **HDR+** (§2.1) — `TileFftMerge` (batched tile FFT + Wiener shrinkage, a genuinely new kernel primitive, not an extension of Phase 1's merge) + exposure-fusion tone mapping via the existing Laplacian-pyramid blend machinery. | Phase 1 (alignment only — its own merge kernel is new) |
| 3 | **Super Res Zoom** (§2.4) — sub-pixel-refine alignment variant, structure tensor, `StructureTensorKernelRegression` accumulation. **Night Sight** (§2.2) then follows almost for free: motion-metering executor (orchestration-level, reuses feature-match) + this phase's merge kernel + its own tone-mapping variant — it is *not* built on Phase 2's HDR+ merge. | Phase 1 (alignment only) |
| 4 | **CLI front end** (§6) — expose everything above headlessly; becomes the primary vehicle for CI regression tests across all burst modes. | Phases 1-3 (incrementally, per subcommand) |
| 5 | **AI inference façade** (§5) — optional ONNX Runtime-backed learned-denoise/learned-SR modes once the classical pipeline is solid. | Phases 1-3 |

## 9. Open questions / risks

- **This repo has no `LICENSE` file.** Needs a decision before consulting
  GPL-3.0 reference code (`ImageStackAlignator`) any further than reading it
  for algorithm understanding — GPL code cannot be copied into a project
  with an incompatible or unset license.
- **Backend coverage matrix** for new ops should be tracked explicitly
  (§4) — a follow-up to actually create the tracking table once Phase 1
  lands its first new op.
- **Numerical tolerance policy** for cross-backend float parity (§7.1)
  needs to be decided once, not per-test.
- **`docs/ARCHITECTURE.md` §1 is now stale**: it states "no CPU-only
  fallback rendering path" and "no AMD/Intel GPU support" as non-goals, but
  the CPU/AVX and Vulkan backends shipped since that doc was written
  directly contradict both. Flagging here; not corrected in this pass —
  worth a dedicated follow-up edit to `ARCHITECTURE.md` §1 so it stops
  reading as current intent.

## 10. Sources

- Hasinoff et al. 2016, *Burst photography for HDR and low-light imaging on
  mobile cameras* —
  [paper PDF](https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus.pdf),
  [supplement](https://static.googleusercontent.com/media/hdrplusdata.org/en//hdrplus_supp.pdf),
  [hdrplusdata.org](https://hdrplusdata.org/) + [dataset](https://hdrplusdata.org/dataset.html) (CC BY-SA 4.0)
- Night Sight — [Google AI blog](https://research.google/blog/night-sight-seeing-in-the-dark-on-pixel-phones/),
  [astrophotography blog](https://research.google/blog/astrophotography-with-night-sight-on-pixel-phones/)
- Wronski et al. 2019, *Handheld Multi-Frame Super-Resolution* —
  [arXiv:1905.03277](https://ar5iv.labs.arxiv.org/html/1905.03277),
  [project page](https://sites.google.com/view/handheld-super-res/)
- [timothybrooks/hdr-plus](https://github.com/timothybrooks/hdr-plus) (MIT, C++/Halide)
- [martin-marek/hdr-plus-pytorch](https://github.com/martin-marek/hdr-plus-pytorch) (MIT)
- [martin-marek/hdr-plus-swift](https://github.com/martin-marek/hdr-plus-swift) (GPL-3.0 — reference only)
- [amonod/hdrplus-python](https://github.com/amonod/hdrplus-python) (AGPL-3.0 — reference only)
- [Jamy-L/Handheld-Multi-Frame-Super-Resolution](https://github.com/Jamy-L/Handheld-Multi-Frame-Super-Resolution) (MIT)
- [kunzmi/ImageStackAlignator](https://github.com/kunzmi/ImageStackAlignator) (GPL-3.0 — reference only)
- [liuzhen03/HDR-Transformer-PyTorch](https://github.com/liuzhen03/HDR-Transformer-PyTorch) (ECCV 2022; license unstated)
- [CharlieMarcotte/Deep-HDR-with-Pytorch](https://github.com/CharlieMarcotte/Deep-HDR-with-Pytorch) (license unstated)
- [google/burst-denoising](https://github.com/google/burst-denoising) (KPN, Apache-2.0, archived/TF1)
- [akshaydudhane16/Burstormer](https://github.com/akshaydudhane16/Burstormer) (MIT)
- [goutamgmb/deep-burst-sr](https://github.com/goutamgmb/deep-burst-sr) (CC-BY-NC-SA — reference only),
  [BSRT](https://github.com/Algolzw/BSRT), BurstSR dataset
  ([arXiv:2101.10997](https://arxiv.org/abs/2101.10997))
- ONNX Runtime execution providers —
  [onnxruntime.ai/docs/execution-providers](https://onnxruntime.ai/docs/execution-providers/)
- [microsoft/DirectML](https://github.com/microsoft/DirectML),
  [Windows ML execution providers](https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/select-execution-providers)
- [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (MIT),
  [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- darktable 5.6 AI subsystem —
  [announcement](https://www.darktable.org/2026/06/meet-darktable-5.6-ai-tools/),
  [docs](https://docs.darktable.org/usermanual/development/en/special-topics/ai/overview/),
  [darktable-org/darktable-ai](https://github.com/darktable-org/darktable-ai)
- darktable OpenCL pixelpipe pattern —
  [activation docs](https://docs.darktable.org/usermanual/4.6/en/special-topics/opencl/activate-opencl/)
- Halide — [CACM research page](https://cacm.acm.org/research/halide/)
- RawTherapee CLI — [rawpedia](https://rawpedia.rawtherapee.com/Command-Line_Options)
- FFmpeg hardware context system —
  [deepwiki overview](https://deepwiki.com/FFmpeg/FFmpeg/7.1-hardware-context-system)
