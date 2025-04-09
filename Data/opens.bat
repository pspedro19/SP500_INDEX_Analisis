@echo off
pause
cd /d "%~dp0"
wsl.exe -d Ubuntu --cd "%CD:\=/%"
