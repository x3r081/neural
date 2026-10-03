@echo off
setlocal
cd /d "%~dp0"
if not defined NEURAL_PYTHON set "NEURAL_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%NEURAL_PYTHON%" (
  echo Neural is not installed yet. Double-click install_neural.bat first.
  pause
  exit /b 1
)
"%NEURAL_PYTHON%" -m neural_runtime start %*
set "NEURAL_EXIT=%errorlevel%"
if not "%NEURAL_EXIT%"=="0" pause
exit /b %NEURAL_EXIT%
