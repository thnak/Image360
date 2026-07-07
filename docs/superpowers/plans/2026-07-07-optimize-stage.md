# Optimize (Stage 2) — Gain, Color Transfer, Bundle Adjustment

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Stage 2 (`docs/ARCHITECTURE.md` §4.3, §7.3) as one
composite `OptimizeExecutor` (registered for `STAGE2_OPTIMIZE`,
dispatching by `task.unitKind`, same pattern `2026-07-07-align-stage.md`
established for `AlignExecutor`) covering three kinds of work: per-image
gain compensation (`unit_kind = "gain"`), per-image color transfer
(`unit_kind = "color"`), and one global, checkpointed bundle-adjustment
task (`unit_kind = "ba_checkpoint"`) — the first real use of
`Task::checkpointJson`.

**Depends on:** `2026-07-07-align-stage.md` (needs completed homographies
and persisted per-image feature blobs), `2026-07-07-raw-ingest.md`
(needs full-res demosaiced blobs for color transfer).

**Honesty note on "unchanged in spirit from v1":** v1's `WorkflowController::Stage2_Optimize`
(deleted in issue #2, still visible via `git show ed75007^:...`) never
actually implemented gain compensation, color transfer, or bundle
adjustment — all three were `TODO`-stubbed to return `true` / default
values. There is no working v1 algorithm to port; this plan designs real
ones from scratch, using the well-known techniques v1's own TODO comments
named as intent (least-squares gain, Reinhard color transfer,
Levenberg-Marquardt bundle adjustment) as the concrete starting point.

**No new JSON dependency:** `checkpoint_json` is a `TEXT` column, not a
type-enforced JSON column — this plan does not vendor a JSON library for
one narrow, fixed-shape payload (`{iteration, lambda, parameters[]}`).
It emits spec-compliant JSON text (inspectable/debuggable with any JSON
tool) via a small purpose-built writer, and reads it back with an
equally narrow, purpose-built parser for exactly that shape — not a
general JSON parser. If a second consumer ever needs general JSON, that's
the point to introduce a real library, not before.

## Global Constraints

- One executor, dispatch-by-`unitKind`, same reasoning as
  `2026-07-07-align-stage.md`'s Task 6 — `TaskScheduler` maps one
  executor per `PipelineStage`.
- Gain/color transfer are ordinary per-image tasks (embarrassingly
  parallel, resumed by re-running non-`COMPLETED` ones — no special
  handling). Bundle adjustment is the **one exception**: a single task
  that resumes by loading its last checkpoint and continuing the LM
  solve, per §7.3 — not re-run from scratch.
- GPU numeric correctness (color transfer's LAB kernels, BA's
  `TensorSolveNormalEquations`-driven solve) cannot be verified from this
  environment — same standing caveat as every GPU-touching plan so far.

---

### Task 1: `OptimizeExecutor` skeleton + task seeding

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/OptimizeExecutor.h`
- Create: `WindowsApp.Core/SourceFiles/OptimizeExecutor.cpp`
- Modify: `WindowsApp.Core/HeaderFiles/ProjectManager.h`
- Modify: `WindowsApp.Core/SourceFiles/ProjectManager.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
class OptimizeExecutor : public ITaskExecutor // registered for STAGE2_OPTIMIZE
{
public:
    OptimizeExecutor(ProjectManager&, StorageEngine&, std::shared_ptr<Compute::CudaPipeline>,
                      std::shared_ptr<Compute::NvJpegCodec>);

    bool Execute(Task& task, CancellationToken token) override; // dispatches by task.unitKind

private:
    bool ExecuteGain(Task& task, CancellationToken token);          // "gain"
    bool ExecuteColorTransfer(Task& task, CancellationToken token); // "color"
    bool ExecuteBundleAdjustment(Task& task, CancellationToken token); // "ba_checkpoint"
};

// ProjectManager.h
bool SeedOptimizeTasks(); // one "gain" + one "color" task per input
                           // image, one single "ba_checkpoint" task
                           // (unit_key = "global" - there is exactly one).
```

- [ ] **Step 1: `Execute` dispatch** (same three-way `if` shape as
  `AlignExecutor::Execute`, default `return false` for an unknown
  `unitKind`).

- [ ] **Step 2: `SeedOptimizeTasks`**

For every input image: `Task{stage=STAGE2_OPTIMIZE, unitKind="gain",
unitKey=imageId}` and `Task{..., unitKind="color", unitKey=imageId}`.
Plus exactly one `Task{stage=STAGE2_OPTIMIZE, unitKind="ba_checkpoint",
unitKey="global"}` — `CreateTasksIfAbsent`'s existing `UNIQUE(stage,
unit_kind, unit_key)` constraint already makes a second call a no-op, so
no extra guard is needed to avoid creating two BA tasks.

- [ ] **Step 3: Add to `.vcxproj`**, header/source consistency check.

---

### Task 2: Gain compensation (`"gain"`)

**Files:**
- Modify: `WindowsApp.Core/SourceFiles/OptimizeExecutor.cpp`

**Approach:** pairwise mean-intensity ratio against a fixed reference
image (image id `0`, the same reference-frame convention
`2026-07-07-align-stage.md` uses for homography) — not a full joint
least-squares solve over all overlapping pairs simultaneously. Documented
simplification (Self-Review), not a hidden gap: a joint solve is strictly
better but needs every pairwise overlap, not just each image's relation
to one reference, and is flagged as follow-up work.

- [ ] **Step 1: Sample intensities at Align's inlier correspondences**

For the target image (not the reference), find its completed Align
`"pair"` task versus the reference image (`GetTasksForStage(STAGE1_ALIGN)`,
`unit_kind == "pair"`, `unit_key` matching either order) — if none
exists (the pair had too few inliers, or wasn't a seeded candidate),
default gain to `1.0` and mark this a normal `false`-returning expected
failure only if no result at all can be produced; **a default of 1.0 is
itself a valid successful result** (returns `true`), not a failure,
matching v1's own "no gain data → assume 1.0" fallback intent.

- [ ] **Step 2: Compute the ratio**

Re-decode both images' embedded previews (`GetEmbeddedPreviewJpeg` +
`NvJpegCodec::Decode`, same calls `AlignExecutor` already makes — no new
Compute-side code needed here), sample luma at the correspondence points
used for that pair's successful RANSAC (re-run `MatchFeatures` against
the two images' persisted feature blobs to recover the same inlier set,
rather than threading it through as new state — see this plan's header
note on why BA does the same re-derivation), compute
`gain = mean(referenceLuma) / mean(targetLuma)` clamped to a sane range
(e.g. `[0.25, 4.0]` — extreme ratios usually indicate a bad match, not a
real exposure difference).

- [ ] **Step 3: Persist**

`m_projectManager.UpdateImageGain(imageId, gain)` (already exists,
Plan 1) — `task.outputBlobId` stays unset; gain isn't blob data, it's a
scalar already living on `input_images`, so this task's "output" is a
side effect on that row, not a new blob. `TaskScheduler`'s commit path
(`item.task.outputBlobId.has_value() ? CommitTaskOutput(...) :
UpdateTaskStatus(..., COMPLETED)`, from `2026-07-07-task-scheduler-core.md`)
already handles a task with no blob output correctly — no `TaskScheduler`
change needed.

---

### Task 3: Color transfer (`"color"`)

**Files:**
- Create: `WindowsApp.Compute/HeaderFiles/color_transfer.cuh`
- Create: `WindowsApp.Compute/SourceFiles/color_transfer.cu`
- Modify: `WindowsApp.Compute/HeaderFiles/CudaPipeline.h`
- Modify: `WindowsApp.Compute/WindowsApp.Compute.vcxproj`
- Modify: `WindowsApp.Core/SourceFiles/OptimizeExecutor.cpp`

**Interfaces:**
```cpp
// color_transfer.cuh, namespace WindowsApp::Compute::Kernels
__global__ void RgbToLabKernel(const unsigned short* __restrict__ rgb, float* __restrict__ lab, int numPixels);
__global__ void LabToRgbKernel(const float* __restrict__ lab, unsigned short* __restrict__ rgb, int numPixels);
// Reduction to per-channel mean/stddev - a standard two-pass (or
// single-pass Welford) parallel reduction, same shared-memory-reduction
// idiom this project's existing kernels don't yet have an example of but
// is standard CUDA practice; implementer picks two-pass-with-atomics for
// simplicity in this first version, noted as not the most efficient
// approach (a proper tree reduction is follow-up work).
__global__ void LabStatsKernel(const float* __restrict__ lab, int numPixels, double* outSum, double* outSumSq);
__global__ void ReinhardTransferKernel(
    float* __restrict__ lab, int numPixels,
    const double srcMean[3], const double srcStd[3],
    const double refMean[3], const double refStd[3]);

// CudaPipeline.h façade
ComputeResult ApplyReinhardColorTransfer(
    unsigned short* rgbInOut, int width, int height,
    const double srcMean[3], const double srcStd[3],
    const double refMean[3], const double refStd[3]);
ComputeResult ComputeLabStats(const unsigned short* rgb, int width, int height, double outMean[3], double outStd[3]);
```

- [ ] **Step 1: RGB↔LAB kernels**

Standard sRGB → CIE LAB conversion (via linear RGB → XYZ → LAB, D65
white point - textbook formulas, not this codebase's invention) and its
inverse.

- [ ] **Step 2: `LabStatsKernel` + `ComputeLabStats` façade**

Per-channel mean and standard deviation over the full image (two-pass:
one kernel launch for sums, host computes mean, a second pass - or
`atomicAdd`-based single pass into `double` accumulators - for the
variance term).

- [ ] **Step 3: `ReinhardTransferKernel` + `ApplyReinhardColorTransfer` façade**

Standard Reinhard transfer: `lab' = (lab - srcMean) * (refStd / srcStd) + refMean`,
per channel, then `LabToRgbKernel` back.

- [ ] **Step 4: `OptimizeExecutor::ExecuteColorTransfer`**

1. Load the target image's full-res demosaiced blob (from `RawIngest`'s
   `STAGE0_INGEST`/`"image"` task's `output_blob_id`,
   `StorageEngine::ReadPixelBuffer`).
2. Same for the reference image (id `0`) - compute its stats once and
   reuse across every other image's color-transfer task rather than
   recomputing per task (an optimization worth a one-line note, not a
   hard requirement for this plan's first pass - recomputing it inside
   every task is also correct, just wasteful; leave as an accepted
   inefficiency, not a bug, until profiling says otherwise).
3. `ComputeLabStats` both, `ApplyReinhardColorTransfer` on the target,
   `StorageEngine::WritePixelBuffer` the result (`formatTag =
   "color_corrected_rgb48"`), set `task.outputBlobId`.
4. The reference image itself still needs a `"color"` task (seeded like
   every other image) - for the reference, this is a no-op passthrough
   (write back the same pixel data unchanged, still through
   `WritePixelBuffer` so `Render` (next plan) can uniformly read "the
   color-transfer output blob" for every image without a reference-image
   special case).

---

### Task 4: Bundle adjustment (`"ba_checkpoint"`)

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/BundleAdjustment.h`
- Create: `WindowsApp.Core/SourceFiles/BundleAdjustment.cpp`
- Modify: `WindowsApp.Core/SourceFiles/OptimizeExecutor.cpp`

**Interfaces:**
```cpp
// BundleAdjustment.h
struct BaCheckpoint
{
    int iteration = 0;
    float lambda = 1e-3f;
    std::vector<float> parameters; // 8 DOF per non-reference image (h00..h21, h22 fixed to 1), flattened, image-id order
};

std::string SerializeCheckpoint(const BaCheckpoint& cp);   // hand-rolled, see this plan's header note
bool DeserializeCheckpoint(const std::string& json, BaCheckpoint& out);

// One LM iteration: builds residuals/Jacobian from every input image's
// current homography (from `parameters`) against Align's persisted
// correspondences, calls CudaPipeline::TensorSolveNormalEquations, applies
// the resulting delta if it reduces total reprojection error (classic LM
// accept/reject + lambda up/down), returns the updated checkpoint and
// whether this iteration converged (delta below a threshold).
struct LmStepResult { BaCheckpoint checkpoint; bool converged; };
LmStepResult RunOneLmIteration(
    Compute::CudaPipeline& cudaPipeline,
    ProjectManager& projectManager,
    const BaCheckpoint& current);
```

- [ ] **Step 1: `SerializeCheckpoint`/`DeserializeCheckpoint`**

Emit `{"iteration":N,"lambda":F,"parameters":[f0,f1,...]}` via
`std::ostringstream`/`snprintf`-style formatting (matches this
codebase's existing preference for straightforward formatted I/O over
new dependencies). `DeserializeCheckpoint` is a narrow scanner for
exactly this shape (find `"iteration":`, read the int; find `"lambda":`,
read the float; find `"parameters":[`, read comma-separated floats until
`]`) - not a general JSON parser, and should reject/return `false` on
anything that doesn't match this exact shape rather than trying to be
lenient.

- [ ] **Step 2: Correspondence re-derivation**

For every input-image pair with a `COMPLETED` Align `"pair"` task,
re-run `MatchFeatures` against their persisted feature blobs (same
re-derivation `2026-07-07-optimize-stage.md`'s Task 2 already does for
gain) to get the correspondence set this iteration's residuals are built
from - computed once per `ExecuteBundleAdjustment` call (not once per
LM iteration inside it - see Step 4), since it doesn't depend on the
current parameter estimate.

- [ ] **Step 3: `RunOneLmIteration`**

Build the reprojection-error residual vector and its Jacobian (standard
BA formulation: for each correspondence, residual = observed point in
image B minus image B's homography applied to the corresponding point in
image A, using the current `parameters`; Jacobian = partial derivatives
of that residual w.r.t. each of the 8 free homography parameters for
image B) on the CPU (this is orchestration/assembly, not itself the
tensor-core-accelerated part - `TensorSolveNormalEquations` is what's
GPU-accelerated), call `TensorSolveNormalEquations(J, r, delta,
numResiduals, numParams, current.lambda)`, apply `delta`, recompute total
reprojection error; if it improved, accept (`lambda /= 10`), else reject
(revert, `lambda *= 10`) — textbook Levenberg-Marquardt damping-factor
adaptation.

- [ ] **Step 4: `OptimizeExecutor::ExecuteBundleAdjustment`**

1. Load the current checkpoint (`task.checkpointJson` via
   `DeserializeCheckpoint`, or a fresh `BaCheckpoint{}` with identity
   parameters if empty - covers both "never started" and "resuming").
2. Loop, calling `RunOneLmIteration` each pass, checking
   `token.stop_requested()` **between** iterations (not mid-iteration -
   matches this plan's standing cancellation convention; one LM
   iteration's GPU work always finishes once started).
3. Every `K` iterations (e.g. 5) or on cancellation, persist via
   `m_projectManager.UpdateTaskCheckpoint(task.taskId,
   SerializeCheckpoint(checkpoint))` - the concrete, first-ever real use
   of that method (added in `2026-07-07-vfp-project-schema.md` but never
   exercised until now).
4. On convergence (`LmStepResult::converged`) or a fixed max-iteration
   budget (e.g. 100), write the final parameters back via
   `ProjectManager::UpdateHomography` for every non-reference image, and
   `return true`.
5. On cancellation before convergence, checkpoint the latest state and
   `return false` - `TaskScheduler` leaves the task `PENDING` (not
   `FAILED`, since it didn't exhaust a retry budget, it was cooperatively
   stopped) via the same requeue path every other cancelled-mid-dispatch
   task takes; the next `RunStage` call resumes from the checkpoint via
   Step 1, continuing the LM solve rather than restarting it - the "resume
   means continue a computation" exception from §7.3.

---

### Task 5: Test coverage

**Files:**
- Create: `WindowsApp.Tests/BundleAdjustmentTests.cpp`
- Modify: `WindowsApp.Tests/WindowsApp.Tests.vcxproj`

- [ ] **Step 1: Checkpoint round-trip**

`SerializeCheckpoint` a `BaCheckpoint` with a few known values,
`DeserializeCheckpoint` it back, assert every field matches exactly -
pure C++, no GPU, fully testable here.

- [ ] **Step 2: Malformed-checkpoint rejection**

`DeserializeCheckpoint` on a handful of deliberately malformed strings
(empty, truncated, wrong key names) asserts `false` and does not crash -
guards the "next `RunStage` after a crash mid-checkpoint-write" scenario
(§7.2's crash-recovery guarantee extended to this task's own resume
data, not just its `TaskStatus`).

- [ ] **Step 3: Try available build** (same standing note as every prior
  plan - report the result, don't claim GPU-side correctness was
  verified from this environment).

## Self-Review

- Spec coverage: implements all three sub-parts of `docs/ARCHITECTURE.md`
  §4.3 (gain, color transfer, bundle adjustment) and §7.3's checkpointing
  exception exactly as specified (single task, periodic `checkpoint_json`
  updates, resume-by-continuing not resume-by-restarting).
- Placeholder scan: no TODO-stubs like v1's - every sub-part has a real,
  if simplified, algorithm (documented above).
- Known gaps carried forward: gain compensation is pairwise-vs-reference,
  not a joint least-squares solve; color transfer's LAB stats
  reduction is a simple two-pass kernel, not a tree reduction; BA's
  residual/Jacobian assembly runs on the CPU per iteration (only the
  normal-equations solve is GPU-accelerated) - all three are correct,
  working v1 implementations, with clearly-named follow-up optimizations
  rather than the "TODO, returns true" placeholders they're replacing.
