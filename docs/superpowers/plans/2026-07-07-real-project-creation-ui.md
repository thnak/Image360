# Real Project-Creation UI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `2026-07-07-ui-progress-nonblocking.md`'s scratch-project
stand-in (`DemoStitchExecutor`, hardcoded `demo_stitch.vfp`, 40 fake
`demo_img_N` tasks) with a real multi-file RAW picker → `input_images`
rows → real `RawIngestExecutor`/`AlignExecutor`/`OptimizeExecutor`/
`RenderExecutor` pipeline, reusing the already-shipped, already-verified
non-blocking/progress/cancel UI scaffolding from issues #12-#16 as-is.

**Depends on:** all four prior GPU-stage plans
(`2026-07-07-raw-ingest.md`, `2026-07-07-align-stage.md`,
`2026-07-07-optimize-stage.md`, `2026-07-07-render-stage.md`) plus
`2026-07-07-nvjpeg-export.md` for the export button.

**What does NOT change:** `MainWindow`'s threading/cancellation/progress-
marshaling code (`std::jthread`/`std::stop_source`/`weak_ref`/
`DispatcherQueue::TryEnqueue`) from issue #15 is untouched — this plan
only changes *what* gets seeded and *which* executors get registered
before `PipelineDriver::Run` is called, exactly the "later plans plug in
real executors without touching the UI thread-marshaling code" promise
`2026-07-07-v2-architecture-roadmap.md` made up front.

## Global Constraints

- The "Panorama Stitch" card's controls (`StitchStartButton`/
  `StitchCancelButton`/`StitchProgressBar`/`StitchStatusText`) are
  reused, not replaced — this plan adds a file-picker step before Start
  becomes meaningful, it doesn't redesign the card.
- Manual verification needs real RAW files, a real GPU, and Visual
  Studio — cannot be done from this environment, same standing note as
  every prior UI-touching plan.

---

### Task 1: `PipelineDriver` seeds Render tasks after Optimize, not upfront

**Files:**
- Modify: `WindowsApp.Core/SourceFiles/PipelineDriver.cpp`

**Why this is needed:** `RenderExecutor`'s `chunk_contributors` (from
`2026-07-07-render-stage.md`) depend on each image's **final** homography
— which isn't settled until `STAGE2_OPTIMIZE` (specifically, bundle
adjustment) completes. Seeding `STAGE3_RENDER` tasks at project-creation
time (alongside Ingest/Align/Optimize, whose task *lists* don't depend on
anything but the input-image set) would compute overlap culling against
identity homographies and silently produce a wrong contributor list.
Ingest/Align/Optimize task lists are seeded once, upfront, by whatever
adds the input images (Task 2 below); Render's seeding must happen after
Optimize, so it belongs inside the stage loop itself, not at project
creation.

- [ ] **Step 1: One targeted addition to `PipelineDriver::Run`'s stage
  loop**

Immediately before the loop iteration that calls `scheduler.RunStage(PipelineStage::STAGE3_RENDER,
...)` (and only for that stage - not the others, which don't need this),
call `projectManager.SeedRenderTasks()`. This is safe and idempotent
whether Optimize just ran in this same call or was already `COMPLETED`
from a prior session (resume case) - `SeedRenderTasks` recomputes overlap
culling and re-runs `SetChunkContributors`/`CreateTasksIfAbsent`, both
already idempotent (`SetChunkContributors` replaces existing rows;
`CreateTasksIfAbsent` no-ops on already-seeded keys).

- [ ] **Step 2: Confirm no other stage's seeding was accidentally moved
  into `PipelineDriver`**

Ingest/Align/Optimize seeding stays a project-creation-time concern
(Task 2), not something `PipelineDriver::Run` does for every stage - only
Render has this ordering dependency. Run:
```powershell
Select-String -Path WindowsApp.Core\SourceFiles\PipelineDriver.cpp -Pattern 'SeedRenderTasks|SeedIngestTasks|SeedAlignTasks|SeedOptimizeTasks'
```
Expected: only `SeedRenderTasks` appears.

---

### Task 2: Multi-file RAW picker → real `input_images` rows

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.cpp`

**Interfaces:** no new named XAML controls - reuses `StitchStartButton`.

- [ ] **Step 1: Replace the demo-task seeding block in
  `StitchStartButton_Click`**

Before creating/opening the project file: `FileOpenPicker` configured for
multiple selection (`PickMultipleFilesAsync`, same picker class
`OpenImageButton_Click` already uses, different call) with RAW extension
filters (`.raf`, `.cr2`, `.nef`, `.arw`, `.dng`, matching
`docs/CLAUDE.md`'s documented supported formats). If the user cancels
(empty result), restore the button states and return without starting a
run - do not fall back to the old demo tasks.

- [ ] **Step 2: Per-file `cfaType` detection + `AddInputImage`**

For each picked file: open a scoped `ImageLoader`, call `UnpackRaw`
(from `2026-07-07-raw-ingest.md`) purely to read `RawPlane::cfaType`
(discarding the CFA data itself - this is metadata-only use of an
already-built method, not new decode logic), then
`m_stitchProject.AddInputImage(filePath, Homography{}, cfaType)` (the
`cfaType`-taking overload from `2026-07-07-raw-ingest.md` Task 1) - every
image starts with an identity homography; Align/Optimize are what
compute real ones.

- [ ] **Step 3: Seed the three upfront stages**

`m_stitchProject.SeedIngestTasks()`, `SeedAlignTasks()`,
`SeedOptimizeTasks()` (Render is deliberately excluded here - see Task 1).

- [ ] **Step 4: Project sizing**

`CreateProject`'s `totalWidth`/`totalHeight` need real values now, not
the demo's fixed `8192, 8192` - compute from the picked images'
metadata (`ImageLoader::GetMetadata`, sum/bounding-box logic is a
reasonable v1 approach: e.g. `totalWidth = max image width * imageCount`
as a rough panorama-canvas upper bound, refined once real homographies
exist - this is a coarse placeholder canvas size, not the final
alignment-derived extent, and should be commented as such). Chunk size:
`RecommendedChunkSize(cudaPipeline->GetGpuInfo().totalMemory)` from
`2026-07-07-render-stage.md` Task 2.

---

### Task 3: Register real executors

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.h`
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.cpp`

**Interfaces:** `MainWindow` gains members for the shared engine objects
each real executor needs (constructed once, not per-run):
```cpp
private:
    std::shared_ptr<Compute::CudaPipeline> m_cudaPipeline;
    std::shared_ptr<Compute::NvJpegCodec> m_nvJpegCodec;
    StorageEngine m_stitchStorage; // paired with m_stitchProject, opened together
```

- [ ] **Step 1: Construct/`Initialize()` the engine objects once**

Lazily on first `StitchStartButton_Click` (or in `MainWindow`'s
constructor - either is fine; lazy avoids paying CUDA init cost if the
user never clicks Start Stitch in a session) - `CudaPipeline::Initialize`,
`NvJpegCodec::Initialize`. On failure (`ComputeResult::NO_GPU` etc.),
show a clear `StitchStatusText` message and don't proceed - this is the
first place in the shipping UI that a missing/incompatible GPU becomes a
real, user-visible failure mode rather than a demo that never touched
CUDA at all.

- [ ] **Step 2: `StorageEngine::Open`**

Alongside `m_stitchProject.CreateProject(...)`, open `m_stitchStorage`
against the same project directory/base name (`2026-07-07-storage-engine.md`).

- [ ] **Step 3: Replace `DemoStitchExecutor` registration**

```cpp
m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE0_INGEST,
    std::make_shared<RawIngestExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline));
m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE1_ALIGN,
    std::make_shared<AlignExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline, m_nvJpegCodec));
m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE2_OPTIMIZE,
    std::make_shared<OptimizeExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline, m_nvJpegCodec));
m_pipelineDriver.RegisterExecutor(PipelineStage::STAGE3_RENDER,
    std::make_shared<RenderExecutor>(m_stitchProject, m_stitchStorage, m_cudaPipeline));
```
`DemoStitchExecutor`'s class definition can stay in the file (harmless,
unused) or be deleted - prefer deleting it, since an unused demo class
sitting next to the real pipeline invites confusion about which one is
live; nothing else in this plan depends on keeping it.

- [ ] **Step 4: Add new headers to `pch.h`**

`RawIngestExecutor.h`, `AlignExecutor.h`, `OptimizeExecutor.h`,
`RenderExecutor.h`, `StorageEngine.h`, `CudaPipeline.h`, `NvJpegCodec.h` -
same pattern as issue #14's original `pch.h` additions.

---

### Task 4: Export button

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml` (one new `Button`,
  e.g. `StitchExportButton`, in the existing Panorama Stitch card)
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.h`
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.cpp`

- [ ] **Step 1: `StitchExportButton`**

`IsEnabled="False"` by default, matching `StitchCancelButton`'s pattern -
enabled only once a run completes successfully (`ok == true` in the
completion handler from issue #15's `StitchStartButton_Click`).

- [ ] **Step 2: `StitchExportButton_Click`**

`FileSavePicker` for the destination `.jpg` path, then
`PanoramaExporter::ExportPreviewJpeg` (`2026-07-07-nvjpeg-export.md`) on
a background thread (same `std::jthread`+dispatcher-marshaling shape as
the main stitch run - reuse the pattern, don't block the UI thread for a
multi-gigapixel composite assembly + encode).

---

### Task 5: Manual verification

- [ ] **Step 1: Try available build** (standing note: expected to need
  Visual Studio/CUDA, unavailable here).
- [ ] **Step 2: Manual check (Visual Studio + real GPU + real RAW files
  required, cannot be done from this environment)**: pick a small batch
  of overlapping RAW photos, run the full pipeline, confirm each stage's
  progress text/percentage advances sensibly, confirm the exported
  preview JPEG opens and looks like a plausible (even if rough, given
  every documented v1-quality-gap in this roadmap) panorama - this is the
  first point in the whole roadmap where an actual visual result exists
  to look at, and where the accumulated "GPU correctness needs
  verification" flags from every prior plan get their first real test.

## Self-Review

- Spec coverage: closes the loop the whole roadmap was building toward -
  a real multi-file project, run through every real stage, producing a
  real exported image, using UI scaffolding that was already proven
  non-blocking/cancellable/resumable against a stub before any of this
  existed.
- Placeholder scan: no stand-ins remain after this plan - `DemoStitchExecutor`
  is explicitly removed (Task 3), not left dangling.
- Known gaps carried forward: canvas sizing before alignment is a coarse
  heuristic (Task 2 Step 4), not derived from real image extents;
  every GPU-numeric-correctness caveat from
  `2026-07-07-raw-ingest.md`/`2026-07-07-align-stage.md`/
  `2026-07-07-optimize-stage.md`/`2026-07-07-render-stage.md` finally
  gets its first real-world exercise here, and this plan's own manual
  verification step is the place that should surface (not silently
  absorb) whatever those flagged risks turn out to mean in practice.
