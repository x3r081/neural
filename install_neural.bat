@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\install_neural.ps1" %*
set "setup_result=%errorlevel%"
if not "%setup_result%"=="0" echo Neural setup failed with exit code %setup_result%.
pause
exit /b %setup_result%
