@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_tracking.ps1" -Interactive %*
if errorlevel 1 pause
