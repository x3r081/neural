@echo off
rem The Qwen experiment has been replaced by the model-aware GPT-OSS-based runtime.
echo This launcher now starts Neural's replacement runtime from neural.local.json.
call "%~dp0start_neural_server.bat" %*
exit /b %errorlevel%
