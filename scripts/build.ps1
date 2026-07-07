#Requires -Version 5.1
<#
.SYNOPSIS
    Builds Image360's WindowsApp.slnx on a Windows machine with Visual Studio
    2022+ (Desktop C++ + UWP/Windows App SDK workloads) and CUDA Toolkit 13.3
    installed.

.DESCRIPTION
    Automates the manual "open a VS Developer shell, then run msbuild" steps
    documented in CLAUDE.md and docs/WINDOWS_BUILD_SETUP.md:
      1. Locates the Visual Studio install via vswhere.exe.
      2. Imports the VS Developer PowerShell environment into THIS process
         (sets VCTargetsPath, PATH for cl.exe/nvcc, etc.) so a plain
         PowerShell prompt can run msbuild without manually launching
         "Developer PowerShell for VS 2022".
      3. Best-effort restores the packages.config-based native NuGet
         dependencies (Windows App SDK, C++/WinRT, WIL, WebView2).
      4. Runs msbuild against WindowsApp.slnx.
      5. Optionally runs WindowsApp.Tests via vstest.console.exe.

    Every run's full console output (everything printed above, plus the
    raw msbuild/nuget/vstest output) is also written to a log file
    (output.log at the repo root by default, overwritten each run) - if
    the script fails or produces an unexpected result, share that file for
    debugging instead of retyping the console output by hand.

    NOTE: this script has been written by reading the project's .vcxproj/
    .wapproj files, not by running an actual Windows/CUDA build (that
    environment isn't available to the assistant that wrote it) - if a
    step fails, please report the exact error (or the log file) back so
    it can be corrected.

.PARAMETER Configuration
    Debug or Release. Default: Debug.

.PARAMETER PackageOnly
    Only (re)builds the "WindowsApp (Package)" wapproj target, matching
    CLAUDE.md's "Build/run just the packaged app" command. Assumes
    WindowsApp.vcxproj and its dependencies were already built once -
    the wapproj has BuildProjectReferences=false, so it will NOT rebuild
    them itself. Omit this switch to build the whole solution instead.

.PARAMETER RunTests
    After a successful build, run WindowsApp.Tests via vstest.console.exe.

.PARAMETER SkipRestore
    Skip the NuGet packages.config restore step (use once packages are
    already restored, to save time on repeat builds).

.PARAMETER LogPath
    Where to write the full run log. Default: output.log at the repo root.
    Overwritten on every run (so it always reflects the most recent
    attempt) unless you pass a different path per run.

.EXAMPLE
    .\scripts\build.ps1
    Full Debug|x64 build of the whole solution. Log written to output.log.

.EXAMPLE
    .\scripts\build.ps1 -Configuration Release -RunTests
    Full Release|x64 build, then run the MSTest suite.

.EXAMPLE
    .\scripts\build.ps1 -PackageOnly -SkipRestore
    Fast repackage after the app itself was already built.
#>
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',

    [switch]$PackageOnly,
    [switch]$RunTests,
    [switch]$SkipRestore,

    [string]$LogPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'output.log'),

    # Used when this script is invoked from another script (e.g.
    # scripts/test.ps1 -Build) that's already writing its own transcript -
    # PowerShell doesn't support nested transcripts, so the caller's log
    # already captures this script's output and starting a second one here
    # would just fail with a noisy (harmless) warning.
    [switch]$SkipLog
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$SlnxPath = Join-Path $RepoRoot 'WindowsApp.slnx'

function Write-Step($message)
{
    Write-Host "`n==> $message" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# Log everything printed by this script - including the raw output of
# msbuild/nuget/vstest, since Start-Transcript captures whatever appears in
# the console, not just Write-Host lines - to $LogPath. Wrapped in try/finally
# so the transcript is always stopped (and the log always flushed to disk)
# whether the build succeeds, fails, or throws partway through.
# ---------------------------------------------------------------------------
$transcriptStarted = $false
if (-not $SkipLog)
{
    try
    {
        Start-Transcript -Path $LogPath -Force -IncludeInvocationHeader | Out-Null
        $transcriptStarted = $true
    }
    catch
    {
        Write-Warning "Could not start logging to '$LogPath': $_. Continuing without a log file."
    }
}

try
{
    if (-not (Test-Path $SlnxPath))
    {
        throw "Could not find WindowsApp.slnx at '$SlnxPath'. Run this script from a checkout of the Image360 repo (scripts/build.ps1)."
    }

    # -----------------------------------------------------------------------
    # 1. Locate Visual Studio and import its Developer environment into this
    #    process. Without this, msbuild fails with MSB4278 ("$(VCTargetsPath)
    #    does not exist") because C++ builds need cl.exe/VCTargetsPath/etc.
    #    on PATH/env, which only a VS install (not a bare .NET SDK) provides.
    # -----------------------------------------------------------------------
    Write-Step "Locating Visual Studio installation"

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere))
    {
        throw "vswhere.exe not found at '$vswhere'. Install Visual Studio 2022+ with the 'Desktop development with C++' workload - see docs/WINDOWS_BUILD_SETUP.md."
    }

    $vsInstallPath = & $vswhere -latest -prerelease -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath

    if (-not $vsInstallPath)
    {
        throw "No Visual Studio installation with the C++ (VC.Tools.x86.x64) component was found. Install the 'Desktop development with C++' workload - see docs/WINDOWS_BUILD_SETUP.md."
    }
    Write-Host "Found VS install: $vsInstallPath"

    $devShellModule = Join-Path $vsInstallPath 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll'
    if (-not (Test-Path $devShellModule))
    {
        throw "Could not find Microsoft.VisualStudio.DevShell.dll under '$vsInstallPath\Common7\Tools'. Your VS install may be incomplete - try repairing it via the Visual Studio Installer."
    }

    Import-Module $devShellModule
    Enter-VsDevShell -VsInstallPath $vsInstallPath -SkipAutomaticLocation `
        -DevCmdArguments '-arch=x64 -host_arch=x64' | Out-Null
    Write-Host "Developer environment imported (VCTargetsPath=$env:VCTargetsPath)"

    # -----------------------------------------------------------------------
    # 2. Sanity-check CUDA Toolkit 13.3 is installed. Not fatal if the check
    #    is inconclusive - msbuild will fail with a clear "CUDA 13.3.props
    #    not found" error later if it's genuinely missing, this is just an
    #    earlier, friendlier signal.
    # -----------------------------------------------------------------------
    Write-Step "Checking for CUDA Toolkit 13.3"

    $cudaPath = $env:CUDA_PATH_V13_3
    if (-not $cudaPath) { $cudaPath = $env:CUDA_PATH }

    if ($cudaPath -and (Test-Path $cudaPath) -and ($cudaPath -match '13\.3'))
    {
        Write-Host "Found CUDA Toolkit 13.3 at: $cudaPath"
    }
    else
    {
        Write-Warning "Could not confirm CUDA Toolkit 13.3 via CUDA_PATH_V13_3/CUDA_PATH (found: '$cudaPath'). If the build fails with a missing 'CUDA 13.3.props'/'CUDA 13.3.targets' error, install CUDA Toolkit 13.3 AFTER Visual Studio (its installer adds the VS build-customization files) - see docs/WINDOWS_BUILD_SETUP.md."
    }

    # -----------------------------------------------------------------------
    # 3. Best-effort NuGet restore for the packages.config-based native
    #    dependencies (Windows App SDK, C++/WinRT, WIL, WebView2) referenced
    #    by WindowsApp/WindowsApp/packages.config.
    # -----------------------------------------------------------------------
    if (-not $SkipRestore)
    {
        Write-Step "Restoring NuGet packages"

        $nugetExe = Join-Path $vsInstallPath 'Common7\IDE\CommonExtensions\Microsoft\NuGet\NuGet.exe'
        if (Test-Path $nugetExe)
        {
            try
            {
                & $nugetExe restore $SlnxPath -NonInteractive
                if ($LASTEXITCODE -ne 0)
                {
                    Write-Warning "nuget restore exited with code $LASTEXITCODE. If the build fails on missing Windows App SDK/C++-WinRT headers, open the solution in Visual Studio once and use Solution Explorer > right-click solution > 'Restore NuGet Packages', then re-run this script with -SkipRestore."
                }
            }
            catch
            {
                Write-Warning "nuget restore failed: $_. Falling back to a manual restore: open the solution in Visual Studio once and use 'Restore NuGet Packages', then re-run this script with -SkipRestore."
            }
        }
        else
        {
            Write-Warning "NuGet.exe not found under the VS install ('$nugetExe'). Restore packages manually once via Visual Studio's 'Restore NuGet Packages', then re-run this script with -SkipRestore."
        }
    }
    else
    {
        Write-Host "Skipping NuGet restore (-SkipRestore)."
    }

    # -----------------------------------------------------------------------
    # 4. Build.
    # -----------------------------------------------------------------------
    Write-Step "Building ($Configuration|x64$(if ($PackageOnly) { ', package only' }))"

    $msbuildArgs = @(
        $SlnxPath,
        "/p:Configuration=$Configuration",
        '/p:Platform=x64',
        '/m',
        '/v:minimal'
    )
    if ($PackageOnly)
    {
        $msbuildArgs += '/t:WindowsApp (Package)'
    }

    & msbuild @msbuildArgs
    if ($LASTEXITCODE -ne 0)
    {
        throw "msbuild failed with exit code $LASTEXITCODE. See docs/WINDOWS_BUILD_SETUP.md's troubleshooting section for common causes (missing CUDA build customization, missing UWP workload, unrestored NuGet packages)."
    }
    Write-Host "Build succeeded." -ForegroundColor Green

    # -----------------------------------------------------------------------
    # 5. Optionally run WindowsApp.Tests.
    # -----------------------------------------------------------------------
    if ($RunTests)
    {
        Write-Step "Running WindowsApp.Tests"

        $vstest = Join-Path $vsInstallPath 'Common7\IDE\Extensions\TestPlatform\vstest.console.exe'
        if (-not (Test-Path $vstest))
        {
            Write-Warning "vstest.console.exe not found at '$vstest'. Run the tests from Visual Studio's Test Explorer instead."
        }
        else
        {
            # Default vcxproj output layout is <ProjectDir>\<Platform>\<Configuration>\.
            # Falling back to a recursive search since this hasn't been
            # verified against an actual build output on this project.
            $testDll = Join-Path $RepoRoot "WindowsApp.Tests\x64\$Configuration\WindowsApp.Tests.dll"
            if (-not (Test-Path $testDll))
            {
                Write-Warning "Expected test DLL not found at '$testDll' - searching the repo for it instead."
                $found = Get-ChildItem -Path $RepoRoot -Recurse -Filter 'WindowsApp.Tests.dll' -ErrorAction SilentlyContinue |
                    Select-Object -First 1
                if ($found) { $testDll = $found.FullName }
            }

            if (Test-Path $testDll)
            {
                & $vstest $testDll
            }
            else
            {
                Write-Warning "Could not locate WindowsApp.Tests.dll anywhere under '$RepoRoot' after the build. Run tests from Visual Studio's Test Explorer instead and report this so the script can be fixed."
            }
        }
    }
}
catch
{
    Write-Host "`nFAILED: $_" -ForegroundColor Red
    throw
}
finally
{
    if ($transcriptStarted)
    {
        Stop-Transcript | Out-Null
        Write-Host "`nFull run log written to: $LogPath" -ForegroundColor Yellow
        Write-Host "Share this file if you need help debugging a failure." -ForegroundColor Yellow
    }
}
