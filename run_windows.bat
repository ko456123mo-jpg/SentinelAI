@echo off
rem ============================================================
rem  SentinelAI - AI Cybersecurity Assistant
rem  Developer: Mohammed Moneer Al-absi (2023050086)
rem  One-click launcher for Windows
rem ============================================================
cd /d "%~dp0"

set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py
%PY% --version >nul 2>nul
if errorlevel 1 (
    echo [!] Python NOT found.
    echo.
    echo     FIX: Install Python 3.13 from:
    echo          https://www.python.org/downloads/release/python-313/
    echo     - Scroll down and download "Windows 64-bit installer"
    echo     - IMPORTANT: check "Add python.exe to PATH" during install!
    echo.
    echo     If typing "python" opens Microsoft Store:
    echo     Settings ^> Apps ^> Advanced app settings ^> App execution aliases
    echo     ^> turn OFF python and python3, then run this file again.
    pause
    exit /b 1
)

rem ---- check 64-bit ----
%PY% -c "import struct; exit(0 if struct.calcsize('P')==8 else 1)" >nul 2>nul
if errorlevel 1 (
    echo [!] 32-bit Python detected. TensorFlow requires 64-bit.
    echo     FIX: reinstall 64-bit Python 3.13 from python.org
    pause
    exit /b 1
)

rem ---- check version 3.10 - 3.13 (TensorFlow does not support 3.14+ yet) ----
%PY% -c "import sys; exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [!] Incompatible Python version detected (need 3.10 - 3.13).
    echo     TensorFlow does NOT support your Python version yet.
    echo.
    echo     FIX: install Python 3.13 from:
    echo          https://www.python.org/downloads/release/python-313/
    echo     (download "Windows 64-bit installer", CHECK "Add python.exe to PATH")
    echo     Then run this file again.
    pause
    exit /b 1
)

rem ---- install requirements on first run ----
%PY% -c "import flask" >nul 2>nul
if errorlevel 1 (
    echo [1/2] First run: installing requirements - please wait 5-10 minutes...
    echo        (about 600 MB, needed once only)
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [!] Installation failed. Check your internet connection and disk space.
        pause
        exit /b 1
    )
)

echo [2/2] Starting SentinelAI web console...
echo.
echo       Browser will open at:  http://localhost:7860
echo       KEEP THIS WINDOW OPEN while using the console.
echo       To stop: press Ctrl+C or close this window.
echo.
%PY% -m src.webapp
pause
