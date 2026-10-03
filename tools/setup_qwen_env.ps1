param([string]$BasePython = 'F:\AI\Neural\.venv\Scripts\python.exe')
$ErrorActionPreference = 'Stop'
$qwenRoot = Split-Path -Parent $PSScriptRoot
$qwenVenv = Join-Path $qwenRoot '.venv'
& $BasePython -m venv $qwenVenv
if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }
# Share installed dependencies read-only; all Qwen code and configuration live
# in this repository. Do not pip-upgrade the original GPT-OSS environment.
$qwenDependencyPath = & $BasePython -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
if ($LASTEXITCODE -ne 0) { throw 'Cannot locate existing dependency environment' }
$qwenPth = Join-Path $qwenVenv 'Lib\site-packages\neural_shared_dependencies.pth'
Set-Content -LiteralPath $qwenPth -Value $qwenDependencyPath -Encoding utf8
& (Join-Path $qwenVenv 'Scripts\python.exe') -c 'import torch, transformers, safetensors; print("Dependencies ready:", torch.__version__, transformers.__version__, safetensors.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed' }
