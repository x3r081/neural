param(
    [string]$Compiler = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repo "kernels\qwen_bf16_experts.c"
if (-not $Output) { $Output = Join-Path $repo "qwen_bf16_experts.dll" }

if (-not $Compiler) {
    $fromPath = Get-Command gcc -ErrorAction SilentlyContinue
    if ($fromPath) {
        $Compiler = $fromPath.Source
    } else {
        $devkit = $env:NEURAL_DEVKIT
        if (-not $devkit) { $devkit = "F:\AI\Neural\third_party\tools\w64devkit\bin" }
        $candidate = Join-Path $devkit "gcc.exe"
        if (-not (Test-Path -LiteralPath $candidate)) {
            throw "gcc was not found on PATH or at $candidate; set NEURAL_DEVKIT or pass -Compiler."
        }
        $Compiler = $candidate
    }
}
if (-not (Test-Path -LiteralPath $source)) { throw "Missing kernel source: $source" }
$outDir = Split-Path -Parent $Output
if ($outDir) { New-Item -ItemType Directory -Force -Path $outDir | Out-Null }

# -march=native targets the local AVX-512 host. Do not copy this DLL to a
# machine with a different CPU instruction set; rebuild it there.
$compilerDir = Split-Path -Parent $Compiler
$oldPath = $env:PATH
if ($compilerDir) { $env:PATH = "$compilerDir;$oldPath" }
& $Compiler -O3 -march=native -fopenmp -shared -o $Output $source
$compileCode = $LASTEXITCODE
$env:PATH = $oldPath
if ($compileCode -ne 0) { throw "gcc failed with exit code $compileCode" }
Write-Output "Built $Output"
