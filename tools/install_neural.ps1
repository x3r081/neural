[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$SkipDependencies,
    [string]$DataDir,
    [string]$ModelDir,
    [string]$RawStoreDir,
    [string]$StoreDir
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repoRoot
$venv = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'

function Test-Python312([string]$Python, [string[]]$Selector) {
    & $Python @Selector -c "import platform,sys; assert sys.version_info[:2] == (3,12) and platform.architecture()[0] == '64bit'"
    return ($LASTEXITCODE -eq 0)
}

$basePython = $null
$baseSelector = @()
$pyLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
if ($pyLauncher) {
    $basePython = $pyLauncher.Source
    $baseSelector = @('-3.12')
    if (-not (Test-Python312 $basePython $baseSelector)) { $basePython = $null }
}
if (-not $basePython) {
    $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pythonCommand -and (Test-Python312 $pythonCommand.Source @())) {
        $basePython = $pythonCommand.Source
        $baseSelector = @()
    }
}
if (-not $basePython -and (Test-Path -LiteralPath $venvPython -PathType Leaf) -and (Test-Python312 $venvPython @())) {
    $basePython = $venvPython
    $baseSelector = @()
}
if (-not $basePython) {
    Write-Error 'Neural setup needs 64-bit Python 3.12. Install it from https://www.python.org/downloads/release/python-31210/ and rerun install_neural.bat.'
    exit 2
}
if ((Test-Path -LiteralPath $venvPython -PathType Leaf) -and -not (Test-Python312 $venvPython @())) {
    Write-Error "The existing .venv is not 64-bit Python 3.12. It was left untouched. Repair it or move it aside before setup; Neural will not replace it."
    exit 2
}

$pathArgs = @()
if ($DataDir) { $pathArgs += @('--data-dir', $DataDir) }
if ($ModelDir) { $pathArgs += @('--model-dir', $ModelDir) }
if ($RawStoreDir) { $pathArgs += @('--raw-store-dir', $RawStoreDir) }
if ($StoreDir) { $pathArgs += @('--store-dir', $StoreDir) }

if ($CheckOnly) {
    $checkPython = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { $basePython }
    $checkSelector = if ($checkPython -eq $basePython) { $baseSelector } else { @() }
    & $checkPython @checkSelector -m neural_runtime.setup --check --skip-dependencies @pathArgs
    exit $LASTEXITCODE
}

$sharedEnv = $false
$cfg = Join-Path $venv 'pyvenv.cfg'
if (Test-Path -LiteralPath $cfg) {
    $sharedEnv = [bool](Select-String -Path $cfg -Pattern '^\s*include-system-site-packages\s*=\s*true\s*$' -Quiet)
}
$sitePackages = Join-Path $venv 'Lib\site-packages'
if (Test-Path -LiteralPath $sitePackages) {
    foreach ($pth in Get-ChildItem -LiteralPath $sitePackages -Filter '*.pth' -File) {
        foreach ($line in Get-Content -LiteralPath $pth.FullName) {
            $entry = $line.Trim()
            if (-not $entry -or $entry.StartsWith('#') -or $entry.StartsWith('import ')) { continue }
            if ([System.IO.Path]::IsPathRooted($entry)) {
                $resolved = [System.IO.Path]::GetFullPath($entry)
            } else {
                $resolved = [System.IO.Path]::GetFullPath((Join-Path $sitePackages $entry))
            }
            $venvPrefix = $venv.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
            if (-not $resolved.StartsWith($venvPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                $sharedEnv = $true
            }
        }
    }
}

$preflightArgs = @('-m', 'neural_runtime.setup', '--dry-run') + $pathArgs
if ($env:LOCALAPPDATA) {
    $dependencyCacheDir = Join-Path $env:LOCALAPPDATA 'pip\Cache'
} elseif ($env:TEMP) {
    $dependencyCacheDir = Join-Path $env:TEMP 'pip-cache'
} else {
    $dependencyCacheDir = $repoRoot
}
$preflightArgs += @('--dependency-dir', $repoRoot, '--dependency-cache-dir', $dependencyCacheDir)
if ($SkipDependencies -or $sharedEnv) { $preflightArgs += '--skip-dependencies' }
& $basePython @baseSelector @preflightArgs
if ($LASTEXITCODE -ne 0) { Write-Error 'Storage preflight failed. No dependency or model downloads were started.'; exit 2 }

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (Test-Path -LiteralPath $venv) {
        Write-Error "Found $venv without its Python interpreter. It was left untouched. Repair or move that incomplete environment, then rerun setup."
        exit 2
    }
    & $basePython @baseSelector -m venv $venv
    if ($LASTEXITCODE -ne 0) { Write-Error 'Could not create the private repository .venv.'; exit 2 }
}

if (-not $SkipDependencies -and -not $sharedEnv) {
    & $venvPython -m pip install -r (Join-Path $repoRoot 'requirements-runtime.txt') --extra-index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { Write-Error 'Dependency installation failed. The model download has not started.'; exit 2 }
} elseif ($sharedEnv) {
    Write-Host 'Shared Python packages were detected. The installer will verify versions and will not run pip.'
}

$setupArgs = @('-m', 'neural_runtime.setup', '--skip-dependencies') + $pathArgs
& $venvPython @setupArgs
exit $LASTEXITCODE
