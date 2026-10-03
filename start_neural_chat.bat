@echo off
setlocal
cd /d "%~dp0"
if not defined NEURAL_PYTHON set "NEURAL_PYTHON=%~dp0.venv\Scripts\python.exe"
"%NEURAL_PYTHON%" chat_client.py --url http://127.0.0.1:8001/v1 %*
exit /b %errorlevel%
