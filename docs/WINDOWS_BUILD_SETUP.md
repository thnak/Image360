# Windows Build Setup

How to set up a Windows machine to build and run Image360, and how to use
`scripts/build.ps1` to automate the steps CLAUDE.md documents manually.

> **Honesty note:** this doc and `scripts/build.ps1` were written by reading
> the project's `.vcxproj`/`.wapproj` files and the CUDA/Windows App SDK
> docs, not by running an actual build - there's no Windows machine or GPU
> available in the environment that produced them. Treat this as a
> best-effort starting point. If a step doesn't match what actually happens
> on your machine, that's useful information — report it back so this can
> be corrected rather than assuming the mismatch is something you did wrong.

## 1. Prerequisites

- Windows 11 (or Windows 10 1909+; Win11 22H2+ recommended - matches
  `docs/ARCHITECTURE.md`'s target for the eventual DirectStorage work).
- An NVIDIA GPU. The code in this repo today has its CUDA code generation
  hardcoded to `compute_89,sm_89` (`WindowsApp.Compute.vcxproj`) — that's
  Ada Lovelace, i.e. an **RTX 40-series** card specifically. You can build
  on any machine with the CUDA Toolkit installed, but to actually *run* the
  compiled GPU kernels you need an RTX 40-series GPU until someone widens
  `CodeGeneration` (tracked as a known limitation in `docs/ARCHITECTURE.md`
  §10/§13).
- ~15-20 GB free disk for Visual Studio + CUDA Toolkit.

## 2. Install Visual Studio 2022 (17.8+)

Using the Visual Studio Installer, install VS2022 (Community, Professional,
or Enterprise) with these workloads/components:

- **Desktop development with C++** (provides `cl.exe`, the C++ MSBuild
  targets under `VCTargetsPath`, and the v143 toolset the `.vcxproj` files
  target).
- **Universal Windows Platform development** (provides
  `Microsoft.DesktopBridge.props`/`.targets`, needed by
  `WindowsApp (Package).wapproj` for MSIX packaging).
- Under "Individual components," confirm **Windows App SDK C++ Templates**
  is checked (usually pulled in automatically by the two workloads above).

Install this *before* CUDA Toolkit — see the note in the next section for
why the order matters.

## 3. Install CUDA Toolkit 13.3

Install CUDA Toolkit version 13.3 from NVIDIA's official CUDA Toolkit
downloads page, choosing the Windows / x86_64 / your Windows version
installer.

**Install order matters:** the CUDA installer detects any already-installed
Visual Studio and adds its MSBuild "build customization" files
(`CUDA 13.3.props`/`CUDA 13.3.targets`) under
`<VS install>\MSBuild\Microsoft\VC\v170\BuildCustomizations\`, which is
exactly what `WindowsApp.Compute.vcxproj` imports by exact version string.
If you install CUDA *before* VS, or install a new VS after CUDA, these
files won't be there and the build fails with an import error naming that
exact file — if that happens, re-run the CUDA installer and choose
"Modify" to reinstall just the Visual Studio integration.

Verify after installing:
```powershell
nvcc --version
```
Should report release 13.3. Also confirm one of `CUDA_PATH` /
`CUDA_PATH_V13_3` is set (the installer sets these automatically) -
`scripts/build.ps1` checks for this and warns (not fails) if it can't
confirm it.

## 4. Restore NuGet packages

`WindowsApp/WindowsApp/packages.config` lists native NuGet dependencies
(Windows App SDK, C++/WinRT, WIL, WebView2) — these use the classic
packages.config restore mechanism, not `PackageReference`/`dotnet restore`.

`scripts/build.ps1` attempts this automatically via the `NuGet.exe` bundled
with Visual Studio. If that fails (untested against a real build — see the
honesty note above), restore manually once:
1. Open `WindowsApp.slnx` in Visual Studio.
2. Solution Explorer → right-click the solution → **Restore NuGet
   Packages**.

Packages only need restoring once (or after `packages.config` changes) —
subsequent builds can pass `-SkipRestore` to the script.

## 5. Build

From a plain PowerShell prompt (not a special "Developer PowerShell" — the
script imports that environment itself):

```powershell
cd path\to\Image360
.\scripts\build.ps1
```

Options:
```powershell
.\scripts\build.ps1 -Configuration Release              # Release instead of Debug
.\scripts\build.ps1 -RunTests                            # also run WindowsApp.Tests
.\scripts\build.ps1 -PackageOnly -SkipRestore            # fast MSIX repackage only
.\scripts\build.ps1 -LogPath C:\logs\try1.log            # write the log somewhere other than output.log
```

### Log file

Every run writes its full console output — every step this doc describes,
plus the raw `msbuild`/`nuget`/`vstest` output — to `output.log` at the
repo root (overwritten each run). If anything fails or looks wrong, **send
that file** instead of retyping the console output by hand; it has
everything needed to diagnose a build issue in one place (which VS/CUDA
install was found, the exact msbuild command line, and its full error
output).

This mirrors the two commands documented in `CLAUDE.md`:
```powershell
msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64
msbuild WindowsApp.slnx /p:Configuration=Debug /p:Platform=x64 /t:"WindowsApp (Package)"
```
but locates and imports the VS Developer environment first, so it works
from any PowerShell prompt without manually launching "Developer PowerShell
for VS 2022" beforehand. **Use the real `msbuild.exe`, not `dotnet
msbuild`** — confirmed 2026-07-21 on a real build: `dotnet msbuild` fails to
resolve `$(VCTargetsPath)` for the vcxproj/wapproj projects even *inside* a
correctly-initialized VS Developer environment (`vcvars64.bat` run first),
so this isn't a "wrong environment" problem as the troubleshooting table
below used to claim — it's specific to invoking through the .NET SDK's
bundled MSBuild at all. `scripts/build.ps1` already gets this right (calls
`msbuild` directly, never `dotnet msbuild`).

### Building from the Visual Studio IDE instead

Just double-click `WindowsApp.slnx`, select `Debug`/`x64` in the toolbar,
and Build Solution (Ctrl+Shift+B). This is the simplest path if the script
hits an issue the first time — VS's own error list is usually more precise
than msbuild's console output for diagnosing a missing workload/component.

## 6. Running tests

`WindowsApp.Tests` is an MSTest **native** (`CppUnitTestFramework`)
project. Use `scripts/test.ps1` rather than `build.ps1 -RunTests` when
you're iterating on tests specifically — it's a separate script so you can
re-run just the suite without rebuilding everything, and it logs to its
own `test-output.log` instead of mixing test output into `output.log`:

```powershell
.\scripts\test.ps1 -Build                              # build, then run everything
.\scripts\test.ps1                                      # re-run against what's already built
.\scripts\test.ps1 -Filter "ClassName=ImageLoaderTests" # run one test class
```

`build.ps1 -RunTests` still works too (it's convenient for a one-shot
"build and verify" pass) — the two overlap deliberately, use whichever
fits what you're doing.

Visual Studio's **Test Explorer** (Test → Test Explorer) is the more
reliable option if either script's auto-discovery of the built test DLL
doesn't find it (that search logic is unverified against a real build
output layout).

**Current test coverage is minimal.** `WindowsApp.Tests/UnitTests.cpp`
today has exactly one placeholder test (`Assert::IsTrue(true)`) — there is
no real coverage yet for `ImageLoader`, `ProjectManager`,
`WorkflowController`, or the CUDA kernels. `scripts/test.ps1` runs
whatever tests exist; it doesn't add any. Writing real tests for the
current (v1) code, versus waiting for the v2 implementation plans in
`docs/superpowers/plans/` (which already specify real test coverage for
the new schema/`TaskScheduler` code as part of their own tasks), is a
separate decision — see the project's plan files for where that's headed.

## 7. Troubleshooting

| Error | Likely cause | Fix |
|---|---|---|
| `MSB4278: $(VCTargetsPath)/Microsoft.Cpp.Default.props does not exist` | Ran `dotnet msbuild` instead of the real `msbuild.exe` — confirmed 2026-07-21 that this fails even inside a correctly-initialized VS Developer environment, it's not an environment problem | Use `scripts/build.ps1` (calls real `msbuild`), or run plain `msbuild` (not `dotnet msbuild`) after opening a "Developer PowerShell for VS 2022" / running `vcvars64.bat` |
| `MSB4019: ... Microsoft.DesktopBridge.props ... was not found` | "Universal Windows Platform development" workload missing | Add it via the Visual Studio Installer, then re-run |
| Import error naming `CUDA 13.3.props`/`.targets` | CUDA Toolkit installed before VS, or a VS repair/update after CUDA removed the integration | Re-run the CUDA 13.3 installer, choose "Modify," reinstall the Visual Studio integration |
| Missing Windows App SDK / C++-WinRT headers (`winrt/Microsoft.*.h` not found) | NuGet packages not restored | Restore manually via VS's "Restore NuGet Packages" (§4), then rebuild |
| App builds but the GPU pipeline reports "no compatible GPU" or crashes on launch | GPU isn't RTX 40-series (Ada/sm_89) — see §1 | Expected today; widening `CodeGeneration` in `WindowsApp.Compute.vcxproj` (tracked in `docs/ARCHITECTURE.md`) is required to support other architectures |
| `vswhere.exe not found` (from the script) | Visual Studio not installed, or installed without the standard VS Installer | Install VS2022 via the official installer (not a portable/offline layout that skips the Installer component) |

If you hit something not on this list, the most useful thing to send back
is `output.log` from the failed run (§5) — this doc and the script can be
corrected from that far more reliably than from guessing.
