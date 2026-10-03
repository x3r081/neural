@echo off
setlocal
cd /d "%~dp0"
if not defined NEURAL_PYTHON set "NEURAL_PYTHON=%~dp0.venv\Scripts\python.exe"
"%NEURAL_PYTHON%" -m neural_runtime serve %*
exit /b %errorlevel%
