@echo off
rem Compatibility alias for the replacement GPT-OSS default.
call "%~dp0start_neural_chat.bat" %*
exit /b %errorlevel%
