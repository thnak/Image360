#Requires -Version 5.1
<#
.SYNOPSIS
    Runs WindowsApp.Tests (MSTest native / CppUnitTestFramework) on a Windows
    machine with Visual Studio 2022+ installed.

.DESCRIPTION
    Companion to scripts/build.ps1, split out so the test suite can be
    re-run on its own (without rebuilding everything) or, with -Build,
    build WindowsApp.Tests and its WindowsApp.Core/WindowsApp.Compute
    dependencies first via scripts/build.ps1. Locates vstest.console.exe
    the same way build.ps1 locates msbuild (via vswhere.exe).

    Every run's full output goes to test-output.log at the repo root
    (overwritten each run, separate from build.ps1's output.log) - share
    that file if a test run needs debugging.

    NOTE: same caveat as build.ps1 - this has been written by reading the
    project's .vcxproj files and vstest.console.exe's documented CLI, not
    verified against a real Windows/CUDA build or an actual test run.

.PARAMETER Configuration
    Debug or Release. Default: Debug.

.PARAMETER Build
    Build WindowsApp.Tests (and its dependencies) first via
    scripts/build.ps1 -RunTests. Omit to just run vstest against whatever
    test binary is already built.

.PARAMETER Filter
    Optional vstest /TestCaseFilter expression, e.g.
    -Filter "ClassName=ImageLoaderTests" to run a subset instead of the
    whole suite.

.PARAMETER LogPath
    Where to write the full run log. Default: test-output.log at repo root.

.EXAMPLE
    .\scripts\test.ps1 -Build
    Build (Debug|x64) then run the full test suite.

.EXAMPLE
    .\scripts\test.ps1
    Just run whatever's already built - fast re-run after fixing a test.

.EXAMPLE
    .\scripts\test.ps1 -Filter "ClassName=ImageLoaderTests"
    Run only one test class.
#>
[CmdletBinding()]
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Configuration = 'Debug',

    [switch]$Build,
    [string]$Filter,

    [string]$LogPath = (Join-Path (Split-Path -Parent $PSScriptRoot) 'test-output.log')
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step($message)
{
    Write-Host "`n==> $message" -ForegroundColor Cyan
}

$transcriptStarted = $false
try
{
    Start-Transcript -Path $LogPath -Force -IncludeInvocationHeader | Out-Null
    $transcriptStarted = $true
}
catch
{
    Write-Warning "Could not start logging to '$LogPath': $_. Continuing without a log file."
}

try
{
    # -------------------------------------------------------------------
    # 1. Optionally build first, reusing build.ps1 rather than duplicating
    #    its VS-locating/NuGet-restore/msbuild logic. Pass -SkipLog since
    #    this script's own transcript already captures build.ps1's output
    #    (PowerShell doesn't support nested transcripts in one process).
    # -------------------------------------------------------------------
    if ($Build)
    {
        Write-Step "Building WindowsApp.Tests via scripts/build.ps1"
        & (Join-Path $PSScriptRoot 'build.ps1') -Configuration $Configuration -SkipRestore -SkipLog
        # build.ps1 throws on failure, which propagates out of this try
        # block and stops this script too - no separate check needed here.
    }

    # -------------------------------------------------------------------
    # 2. Locate Visual Studio, purely to find vstest.console.exe (no
    #    Developer-shell environment import needed just to run tests).
    # -------------------------------------------------------------------
    Write-Step "Locating vstest.console.exe"

    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere))
    {
        throw "vswhere.exe not found at '$vswhere'. Install Visual Studio 2022+ - see docs/WINDOWS_BUILD_SETUP.md."
    }

    $vsInstallPath = & $vswhere -latest -prerelease -products * -property installationPath
    if (-not $vsInstallPath)
    {
        throw "No Visual Studio installation found via vswhere."
    }

    $vstest = Join-Path $vsInstallPath 'Common7\IDE\Extensions\TestPlatform\vstest.console.exe'
    if (-not (Test-Path $vstest))
    {
        throw "vstest.console.exe not found at '$vstest'. Install the 'Desktop development with C++' workload (it includes the native/C++ test adapter), or run tests from Visual Studio's Test Explorer instead."
    }
    Write-Host "Found: $vstest"

    # -------------------------------------------------------------------
    # 3. Locate the built test DLL.
    # -------------------------------------------------------------------
    Write-Step "Locating WindowsApp.Tests.dll"

    # Default vcxproj output layout is <ProjectDir>\<Platform>\<Configuration>\.
    # Falling back to a recursive search since this hasn't been verified
    # against an actual build output on this project.
    $testDll = Join-Path $RepoRoot "WindowsApp.Tests\x64\$Configuration\WindowsApp.Tests.dll"
    if (-not (Test-Path $testDll))
    {
        Write-Warning "Expected test DLL not found at '$testDll' - searching the repo for it instead."
        $found = Get-ChildItem -Path $RepoRoot -Recurse -Filter 'WindowsApp.Tests.dll' -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) { $testDll = $found.FullName }
    }

    if (-not (Test-Path $testDll))
    {
        throw "Could not find WindowsApp.Tests.dll anywhere under '$RepoRoot'. Build it first: .\scripts\test.ps1 -Build (or build WindowsApp.Tests from Visual Studio)."
    }
    Write-Host "Found: $testDll"

    # -------------------------------------------------------------------
    # 4. Run.
    # -------------------------------------------------------------------
    Write-Step "Running tests$(if ($Filter) { " (filter: $Filter)" })"

    $vstestArgs = @($testDll)
    if ($Filter) { $vstestArgs += "/TestCaseFilter:$Filter" }

    & $vstest @vstestArgs
    if ($LASTEXITCODE -ne 0)
    {
        throw "vstest.console.exe exited with code $LASTEXITCODE - one or more tests failed, or a test host crashed. Scroll up (or check $LogPath) for which test(s)."
    }
    Write-Host "All tests passed." -ForegroundColor Green
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
