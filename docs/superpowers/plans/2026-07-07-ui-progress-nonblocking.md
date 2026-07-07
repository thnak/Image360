# UI: Non-Blocking Pipeline Run With Live Progress & Cancel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `MainWindow` to `PipelineDriver` (from
`2026-07-07-task-scheduler-core.md`) so starting a pipeline run **never
blocks the UI thread**, progress is shown live, and a Cancel button
requests cooperative cancellation — matching `docs/ARCHITECTURE.md` §14.2
exactly. This is also the **first time the UI project links against
`WindowsApp.Core`** at all (today `WindowsApp.vcxproj` has no
`ProjectReference` to it — only `WindowsApp.Tests` does).

Uses `2026-07-07-task-scheduler-core.md`'s `StubTaskExecutor`-backed
`PipelineDriver`, not a real GPU pipeline (RawIngest/Align/Optimize/Render
don't exist yet). This is intentional: it proves out the concurrency and
progress-marshaling mechanics — the actual "does the UI stay responsive
during real work" question — independently of whether the GPU engine is
built yet. A later plan swaps the stub for real executors without touching
this plan's threading/marshaling code.

**Depends on:** `2026-07-07-task-scheduler-core.md`.

**Architecture:** `MainWindow` owns a `std::stop_source` and a
`std::jthread` that runs `PipelineDriver::Run` for the whole project
duration. `PipelineDriver`'s progress callback fires on that background
thread and must never touch XAML directly — it hops back to the UI thread
via the window's `DispatcherQueue::TryEnqueue` before updating any control.
Window close during an active run needs no special handling:
`std::jthread`'s destructor requests stop and joins automatically.

This adds a new, separate "Panorama Stitch" section to `MainWindow.xaml`
alongside the existing "Photo Lab" editor section (from
`docs/superpowers/plans/2026-07-06-image-editor-screen.md`) — it does not
touch or replace the existing editor controls (`OpenImageButton`,
`BrightnessSlider`, etc.).

**Tech Stack:** C++/WinRT, WinUI 3 (`ProgressBar`, `Button`, `TextBlock`),
`Microsoft::UI::Dispatching::DispatcherQueue` (already included via
`pch.h`), `std::jthread`/`std::stop_source` (new includes needed).

## Global Constraints

- Do not modify the existing "Photo Lab" editor controls or their event
  handlers.
- The pipeline run in this slice operates against a scratch/demo project
  seeded with stub tasks (via `ProjectManager`/`TaskScheduler` from prior
  plans) — building real project-creation UX (multi-file RAW picker →
  `input_images` table) is out of scope here; see
  `2026-07-07-v2-architecture-roadmap.md` for where that lands.
- Full build may require Visual Studio C++/CUDA toolchains not available to
  `dotnet msbuild` in a plain shell — verification notes this. Manual
  interactive verification (does the UI actually stay responsive) can only
  be done by running the packaged app in Visual Studio, not from this
  shell — call this out explicitly rather than claiming it was verified.

---

### Task 1: Wire `WindowsApp` to `WindowsApp.Core`

**Files:**
- Modify: `WindowsApp/WindowsApp/WindowsApp.vcxproj`

**Interfaces:**
- N/A (build wiring only)

- [ ] **Step 1: Add the project reference**

Mirror `WindowsApp.Tests.vcxproj`'s existing reference:
```xml
<ItemGroup>
  <ProjectReference Include="..\..\WindowsApp.Core\WindowsApp.Core.vcxproj">
    <Project>{d6b1d440-6228-4444-a1a1-9a9eb2d12b11}</Project>
  </ProjectReference>
</ItemGroup>
```
Also add `WindowsApp.Core`'s and `WindowsApp.Compute`'s header directories
to `AdditionalIncludeDirectories` for `ClCompile`, matching the pattern
`WindowsApp.Core.vcxproj` itself uses to reach `WindowsApp.Compute`'s
headers.

- [ ] **Step 2: Flag the runtime dependency this introduces**

`WindowsApp.Core` is a static lib but transitively depends on
`WindowsApp.Compute`, a **dynamic** lib (CUDA DLL, per
`docs/ARCHITECTURE.md`'s existing description of the project layout). Note
in a build comment or the PR description (not required to solve fully in
this task) that `WindowsApp (Package)`'s `.wapproj` will need
`WindowsApp.Compute.dll` copied into the package output — verify this in
Task 5's manual run, don't assume MSBuild handles it automatically for a
`.wapproj` the way it would for a plain `.exe` project.

- [ ] **Step 3: Confirm the reference resolves**

Run:
```powershell
Select-String -Path WindowsApp\WindowsApp\WindowsApp.vcxproj -Pattern 'WindowsApp.Core.vcxproj'
```
Expected: match found.

---

### Task 2: XAML — Panorama Stitch section

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml`

**Interfaces:**
- Produces: named controls `StitchStartButton`, `StitchCancelButton`,
  `StitchProgressBar`, `StitchStatusText`.

- [ ] **Step 1: Add a new card below the existing "Tools"/"Layers" cards**

Same visual style as the existing `Border`/`CornerRadius="8"` cards in the
right-hand `ScrollViewer` column. Contents:
- `TextBlock` titled "Panorama Stitch" (`SubtitleTextBlockStyle`, matching
  the existing "Adjustments"/"Tools"/"Layers" headers).
- `TextBlock x:Name="StitchStatusText"` — idle message by default (e.g.
  "No stitch running.").
- `ProgressBar x:Name="StitchProgressBar" Minimum="0" Maximum="100"
  Value="0"`.
- Two buttons side by side: `Button x:Name="StitchStartButton"
  Content="Start Stitch" Click="StitchStartButton_Click"
  Style="{StaticResource AccentButtonStyle}"` and `Button
  x:Name="StitchCancelButton" Content="Cancel" IsEnabled="False"
  Click="StitchCancelButton_Click"`.

- [ ] **Step 2: Parse check**

Run:
```powershell
[xml](Get-Content -Raw WindowsApp\WindowsApp\MainWindow.xaml)
```
Expected: XML parses without errors.

- [ ] **Step 3: Confirm named controls are present**

Run:
```powershell
Select-String -Path WindowsApp\WindowsApp\MainWindow.xaml -Pattern 'StitchStartButton|StitchCancelButton|StitchProgressBar|StitchStatusText'
```
Expected: each name appears.

---

### Task 3: pch.h additions

**Files:**
- Modify: `WindowsApp/WindowsApp/pch.h`

**Interfaces:**
- N/A

- [ ] **Step 1: Add STL threading/cancellation headers**

```cpp
#include <thread>
#include <stop_token>
```
`Microsoft.UI.Dispatching.h` and `wil/cppwinrt_helpers.h` are already
included — no new WinRT headers are needed for dispatcher marshaling.

- [ ] **Step 2: Include the Core engine headers needed by `MainWindow`**

```cpp
#include "PipelineDriver.h"
#include "ProjectManager.h"
#include "TaskScheduler.h"
```
(paths resolved via Task 1's `AdditionalIncludeDirectories`).

---

### Task 4: `MainWindow` wiring

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.h`
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.cpp`

**Interfaces:**
- Consumes: named XAML controls from Task 2; `PipelineDriver`,
  `ProjectManager`, `TaskScheduler`, `StubTaskExecutor`-equivalent seed
  logic (a small private helper, not the test-only `StubTaskExecutor`
  itself — see Step 2) from prior plans.
- Produces: event handlers `StitchStartButton_Click`,
  `StitchCancelButton_Click`.

- [ ] **Step 1: Declare handlers and members in `MainWindow.xaml.h`**

```cpp
void StitchStartButton_Click(
    winrt::Windows::Foundation::IInspectable const& sender,
    winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

void StitchCancelButton_Click(
    winrt::Windows::Foundation::IInspectable const& sender,
    winrt::Microsoft::UI::Xaml::RoutedEventArgs const& args);

private:
    std::stop_source m_stitchStopSource;
    std::jthread m_stitchThread;
    WindowsApp::Core::ProjectManager m_stitchProject;
    WindowsApp::Core::PipelineDriver m_pipelineDriver;
```

`m_stitchThread`'s destructor requests stop and joins automatically if
`MainWindow` is torn down mid-run — no explicit shutdown handler needed for
that case (call this out in a one-line comment so it isn't "fixed" later
under the mistaken impression it's a gap).

- [ ] **Step 2: Implement `StitchStartButton_Click`**

1. Disable `StitchStartButton`, enable `StitchCancelButton`.
2. Create (or reopen, if it already exists from a previous run in this
   session) a scratch project via `m_stitchProject.CreateProject(...)` at a
   fixed path under the app's local folder — this is a deliberately minimal
   stand-in for real project creation (see this plan's Global Constraints).
3. Seed a small number of demo tasks into one stage via
   `CreateTasksIfAbsent`, and register a locally-defined executor for that
   stage on `m_pipelineDriver` — a small in-file class equivalent to
   `StubTaskExecutor` but living in production code, not
   `WindowsApp.Tests` (the test-only header must not be linked into the
   shipping UI binary).
4. Reset `m_stitchStopSource` to a fresh `std::stop_source`.
5. Capture `DispatcherQueue()` into a local, call
   `m_pipelineDriver.Initialize(progressCallback, logCallback)` where
   `progressCallback` captures the dispatcher and does:
   ```cpp
   dispatcher.TryEnqueue([this, stage, progress]
   {
       StitchStatusText().Text(StageToDisplayString(stage));
       StitchProgressBar().Value(progress * 100.0);
   });
   ```
6. Start `m_stitchThread` running
   `m_pipelineDriver.Run(m_stitchProject, m_stitchStopSource.get_token())`;
   on completion (still on the background thread), marshal a final
   dispatcher update that re-enables `StitchStartButton`, disables
   `StitchCancelButton`, and sets `StitchStatusText` to a
   completed/cancelled/failed message based on `Run`'s return value.

- [ ] **Step 3: Implement `StitchCancelButton_Click`**

```cpp
m_stitchStopSource.request_stop();
StitchStatusText().Text(L"Cancelling — finishing in-flight work...");
```
Do not join the thread here (that would block the UI thread) — the
completion handling in Step 2.6 (already running on the background thread,
dispatched back to the UI thread on finish) is what re-enables the Start
button once `Run` actually returns.

- [ ] **Step 4: Confirm handler wiring**

Run:
```powershell
Select-String -Path WindowsApp\WindowsApp\MainWindow.xaml,WindowsApp\WindowsApp\MainWindow.xaml.h,WindowsApp\WindowsApp\MainWindow.xaml.cpp -Pattern 'StitchStartButton_Click|StitchCancelButton_Click'
```
Expected: XAML references, header declarations, and C++ definitions are
all present.

---

### Task 5: Verification

**Files:**
- Verify: all files touched by Tasks 1-4.

- [ ] **Step 1: Try available build**

Run:
```powershell
dotnet msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64 /v:minimal
```
Expected in this shell: likely fails for missing Visual Studio C++/CUDA
toolchains, as with every native build in this repo. If it fails only for
missing targets/toolset, report that and ask the user to build in Visual
Studio.

- [ ] **Step 2: Manual interactive verification (Visual Studio required)**

This cannot be done from a plain shell — say so explicitly rather than
claiming it was verified. Ask the user to, in Visual Studio:
1. Run the packaged app (`WindowsApp (Package)`).
2. Click "Start Stitch" — confirm the window remains fully interactive
   (resize it, interact with the existing "Photo Lab" editor controls)
   while `StitchProgressBar` advances.
3. Click "Cancel" mid-run — confirm the status text updates and the Start
   button re-enables once in-flight stub tasks finish, without the UI
   ever freezing.
4. Start a run and close the window mid-run — confirm the app exits
   cleanly (no hang), verifying `std::jthread`'s automatic
   stop-and-join-on-destruction path.

## Self-Review

- Spec coverage: satisfies "the UI should show progress, do not block UI"
  directly — Task 4 Step 2/3 is the concrete mechanism, Task 5 Step 2 is
  how to confirm it actually holds in the running app, not just in code
  review.
- Placeholder scan: the scratch-project seeding in Task 4 Step 2.2-2.3 is
  explicitly named as a stand-in (not a placeholder) with a stated
  successor (`2026-07-07-v2-architecture-roadmap.md`'s real project
  creation phase).
- Type consistency: control names in XAML match handler signatures in
  `MainWindow.xaml.h`/`.cpp`; `PipelineDriver`/`ProjectManager` usage
  matches the signatures fixed in `2026-07-07-task-scheduler-core.md`.
