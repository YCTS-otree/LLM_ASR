@echo off
cd /d "%~dp0"
set "ASR_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%ASR_PYTHON%" set "ASR_PYTHON=%LocalAppData%\Programs\Python\Python310\python.exe"
if not exist "%ASR_PYTHON%" set "ASR_PYTHON=python"
"%ASR_PYTHON%" app.py
if errorlevel 1 pause
