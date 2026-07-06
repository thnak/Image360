# Image Editor Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first usable image editor screen that can select a picture, preview it, expose Photoshop-style edit controls, and show a GPU-ready preview status.

**Architecture:** Keep the first slice inside the existing `MainWindow` because the starter app has one window and no navigation shell. Use XAML for the editor layout and C++/WinRT event handlers for image selection, preview loading, slider state, and reset behavior. Reserve a named preview surface and status text for a later Direct2D/Direct3D GPU renderer without adding a graphics dependency in this slice.

**Tech Stack:** C++/WinRT, WinUI 3, Windows App SDK storage pickers, XAML `Image` plus `BitmapImage`.

## Global Constraints

- Keep the implementation in the existing C++/WinUI app.
- Do not introduce a new GPU/image library in the first slice.
- Use Windows App SDK `FileOpenPicker` with the current `AppWindow().Id()`.
- Use common image extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.tif`, `.tiff`.
- Keep controls interactive even before a real GPU backend exists.
- Full build may require Visual Studio C++/DesktopBridge targets that are not available to `dotnet msbuild` in this shell.

---

### Task 1: Editor Layout

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml`

**Interfaces:**
- Produces: named controls `OpenImageButton`, `ResetEditsButton`, `EditedImage`, `EditorStatusText`, `BrightnessSlider`, `ContrastSlider`, `SaturationSlider`, `SharpnessSlider`, `GpuPreviewToggle`, and `EffectModeComboBox`.

- [x] **Step 1: Write static structure check**

Run after editing:

```powershell
Select-String -Path WindowsApp\WindowsApp\MainWindow.xaml -Pattern 'OpenImageButton|EditedImage|BrightnessSlider|GpuPreviewToggle|EffectModeComboBox'
```

Expected: each name appears in the XAML.

- [x] **Step 2: Replace the sample component gallery with an editor shell**

Create a two-column editor with a toolbar, large checkerboard preview well, and right-side adjustment panel. Add click and value changed event handlers that Task 2 implements.

- [x] **Step 3: Parse XAML**

Run:

```powershell
[xml](Get-Content -Raw WindowsApp\WindowsApp\MainWindow.xaml)
```

Expected: XML parses without errors.

### Task 2: Picker And UI State

**Files:**
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.h`
- Modify: `WindowsApp/WindowsApp/MainWindow.xaml.cpp`
- Modify: `WindowsApp/WindowsApp/pch.h`

**Interfaces:**
- Consumes: named XAML controls from Task 1.
- Produces: event handlers `OpenImageButton_Click`, `ResetEditsButton_Click`, and `AdjustmentSlider_ValueChanged`.

- [x] **Step 1: Add required WinRT includes**

Add storage picker, imaging, and value-changed headers to `pch.h`.

- [x] **Step 2: Declare event handlers**

Declare the three event handlers in `MainWindow.xaml.h`.

- [x] **Step 3: Implement image selection**

Use `Microsoft::Windows::Storage::Pickers::FileOpenPicker picker(AppWindow().Id())`, append image extensions, await `PickSingleFileAsync`, and set `EditedImage().Source()` to a `BitmapImage` created from the selected path.

- [x] **Step 4: Implement reset and slider status**

Reset sliders to neutral values and update `EditorStatusText` with clear messages when edits change.

### Task 3: Verification

**Files:**
- Verify: `WindowsApp/WindowsApp/MainWindow.xaml`
- Verify: `WindowsApp/WindowsApp/MainWindow.xaml.h`
- Verify: `WindowsApp/WindowsApp/MainWindow.xaml.cpp`
- Verify: `WindowsApp/WindowsApp/pch.h`

- [x] **Step 1: Parse XAML**

Run:

```powershell
[xml](Get-Content -Raw WindowsApp\WindowsApp\MainWindow.xaml)
```

Expected: XML parses without errors.

- [x] **Step 2: Confirm handler declarations and definitions**

Run:

```powershell
Select-String -Path WindowsApp\WindowsApp\MainWindow.xaml,WindowsApp\WindowsApp\MainWindow.xaml.h,WindowsApp\WindowsApp\MainWindow.xaml.cpp -Pattern 'OpenImageButton_Click|ResetEditsButton_Click|AdjustmentSlider_ValueChanged'
```

Expected: XAML references, header declarations, and C++ definitions are present.

- [x] **Step 3: Try available build**

Run:

```powershell
dotnet msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64 /v:minimal
```

Expected in this shell: may fail because Visual Studio C++/DesktopBridge targets are unavailable. If it fails for missing targets only, report that and ask the user to build in Visual Studio.

## Self-Review

- Spec coverage: the plan covers picture selection, an editor-like screen, editable controls, and a GPU-ready status path.
- Placeholder scan: no placeholder steps remain.
- Type consistency: XAML names and C++ handler names match across tasks.
