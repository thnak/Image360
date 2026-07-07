# nvJPEG Export Path — Preview/Share Encode

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `docs/ARCHITECTURE.md` §4.5's second integration
point — GPU-side JPEG encode of the finished panorama for preview/share
export, independent of the archival full-resolution render output (which
stays a lossless format and is untouched by this plan).

**Depends on:** `2026-07-07-align-stage.md` (needs `NvJpegCodec`, decode
side already built there), `2026-07-07-render-stage.md` (needs completed
render chunks to assemble from).

**Not a pipeline stage:** unlike Ingest/Align/Optimize/Render, export is
a user-triggered, on-demand action ("Export preview JPEG"), not one of
`PipelineStage`'s four stages — it has no `ITaskExecutor`/`Task` rows of
its own. It runs once a project's `STAGE3_RENDER` tasks are all
`COMPLETED`, reads their output blobs directly, and writes a plain `.jpg`
file to a user-chosen path outside the project container (not through
`StorageEngine`/`blob_directory` — this is an exported artifact, not
internal pipeline state).

## Global Constraints

- Encode correctness (as with every other GPU-touching plan) can't be
  verified from this environment.
- This plan does not implement NVDEC/NVENC or nvCOMP — explicitly out of
  scope per `docs/ARCHITECTURE.md` §4.5 ("no 360 video export planned
  yet").

---

### Task 1: `NvJpegCodec::Encode`

**Files:**
- Modify: `WindowsApp.Compute/HeaderFiles/NvJpegCodec.h`
- Modify: `WindowsApp.Compute/SourceFiles/NvJpegCodec.cpp`

**Interfaces:**
```cpp
// Encodes an interleaved RGB8 buffer to JPEG bytes. `quality` 0-100,
// matches nvjpegEncoderParamsSetQuality's own range.
ComputeResult Encode(const unsigned char* rgb, int width, int height, int quality,
                      std::vector<unsigned char>& outJpegBytes);
```

- [ ] **Step 1: Implement via `nvjpegEncoderParams`/`nvjpegEncodeImage`**

Standard nvJPEG encode sequence: `nvjpegEncoderParamsCreate` →
`nvjpegEncoderParamsSetQuality`/`SetSamplingFactors` (4:2:0 is a
reasonable default for a preview/share export, not the archival path) →
`nvjpegEncoderStateCreate` → upload `rgb` to device memory →
`nvjpegEncodeImage` (input format `NVJPEG_INPUT_RGBI`) →
`nvjpegEncodeRetrieveBitstream` (called twice, per the nvJPEG API's own
convention: once to get the size, once to fill the buffer) into
`outJpegBytes`.

- [ ] **Step 2: Header/source consistency check.**

---

### Task 2: `PanoramaExporter`

**Files:**
- Create: `WindowsApp.Core/HeaderFiles/PanoramaExporter.h`
- Create: `WindowsApp.Core/SourceFiles/PanoramaExporter.cpp`
- Modify: `WindowsApp.Core/WindowsApp.Core.vcxproj`

**Interfaces:**
```cpp
class PanoramaExporter
{
public:
    PanoramaExporter(ProjectManager&, StorageEngine&, std::shared_ptr<Compute::NvJpegCodec>);

    // maxDimension: the exported preview's longest edge in pixels (a
    // multi-gigapixel archival render is never exported at full
    // resolution as a JPEG preview - this is the whole point of this
    // being a separate, smaller "preview/share" path per SS4.5).
    // Returns false (expected failure - e.g. not every chunk COMPLETED
    // yet, or the destination path isn't writable) rather than throwing.
    bool ExportPreviewJpeg(const std::wstring& destPath, int maxDimension, int quality = 90);
};
```

- [ ] **Step 1: Precondition check**

Every `STAGE3_RENDER` task must be `COMPLETED` (`GetTasksForStage`,
assert none are non-`COMPLETED`) - return `false` immediately otherwise,
with a clear reason (not a partial/corrupt export).

- [ ] **Step 2: Assemble a downsampled composite**

For each chunk: `ReadPixelBuffer` its render output blob, compute its
placement + downsample factor in the target `maxDimension`-bounded
canvas (`scale = maxDimension / max(totalWidth, totalHeight)`), nearest-
neighbor or simple box-downsample blit into one host-side RGB8 buffer
sized to the scaled canvas (16-bit → 8-bit conversion via a straight
`>> 8` shift, matching how this codebase already treats RGB48→display
conversions elsewhere - verify against existing UI preview code for
consistency rather than inventing a new tone-mapping step here, out of
scope for this plan). This composite assembly is plain CPU work (no CUDA
needed for a preview-resolution blit) - keep it simple.

- [ ] **Step 3: Encode + write**

`m_nvJpegCodec->Encode(composite, ..., quality, jpegBytes)`, write
`jpegBytes` to `destPath` via `CreateFileW`/`WriteFile` (plain Win32, not
`StorageEngine` - see this plan's header note on why).

- [ ] **Step 4: Add to `.vcxproj`**, header/source consistency check.

- [ ] **Step 5: Test coverage**

The composite-assembly placement/scale math (Step 2) is pure arithmetic -
testable without GPU: given a few synthetic chunk positions/sizes and a
target `maxDimension`, assert the computed scale factor and each chunk's
placement rectangle in the output canvas are correct. Encode correctness
itself needs real hardware, same standing caveat as every GPU-touching
plan.

## Self-Review

- Spec coverage: implements §4.5's export-encode integration point,
  explicitly scoped to preview/share (not archival) output, matching
  "the finished stitched panorama's preview/share exports... are encoded
  via nvjpegEncodeImage."
- Placeholder scan: no placeholders - both the encode façade and the
  composite assembly are complete, working implementations, not stubs.
- Known gaps carried forward: 16-bit→8-bit tone mapping is a straight
  bit-shift, not a real tone-mapping curve - acceptable for a preview
  export, flagged as a possible quality improvement later, not a
  correctness bug.
