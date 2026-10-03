@echo off
rem Compatibility alias: the common launcher selects the verified fast GPT-OSS profile.
call "%~dp0start_neural_server.bat" %*
exit /b %errorlevel%
