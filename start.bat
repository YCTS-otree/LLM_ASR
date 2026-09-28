@echo off
cd /d "%~dp0"
set "ASR_PYTHON=G:\Python\Python_Environment\Python310\python.exe"
if not exist "%ASR_PYTHON%" (
    echo Python not found: %ASR_PYTHON%
    pause
    exit /b 1
)
"%ASR_PYTHON%" app.py
if errorlevel 1 pause
