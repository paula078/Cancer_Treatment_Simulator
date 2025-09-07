@echo off
title Cancer Treatment Simulator - Firewall Friendly Launcher  
color 0B
echo.
echo ================================================================================
echo CANCER TREATMENT SIMULATOR - FIREWALL FRIENDLY LAUNCHER
echo ================================================================================
echo Using 127.0.0.1 to avoid Windows Firewall issues
echo ================================================================================
echo.

REM Navigate to the script directory
cd /d "%~dp0"

echo 🔍 Setting up firewall-friendly launch...

REM Use virtual environment if available
if exist ".cancer_simulator_env\Scripts\python.exe" (
    echo ✅ Using virtual environment Python...
    set PYTHON_CMD=".cancer_simulator_env\Scripts\python.exe"
) else (
    echo ⚠️  No virtual environment found, using system Python...
    set PYTHON_CMD=py -3.10
)

REM Find a free port starting from 8502
set PORT=8502
:FINDPORT
netstat -an | find ":%PORT%" >nul
if not errorlevel 1 (
    set /a PORT+=1
    if %PORT% GTR 8510 (
        echo ❌ No free ports found between 8502-8510
        goto ERROR
    )
    goto FINDPORT
)

echo ✅ Using port %PORT%
echo.
echo 🚀 Starting Cancer Treatment Simulator...
echo.
echo 🌐 Open in browser: http://127.0.0.1:%PORT%
echo                   or: http://localhost:%PORT%
echo.
echo 💡 This launcher uses 127.0.0.1 to avoid firewall issues
echo 💡 Close this window to stop the application
echo.

REM Launch with 127.0.0.1 instead of localhost
%PYTHON_CMD% -m streamlit run main.py --server.port %PORT% --server.address 127.0.0.1 --server.headless false

goto END

:ERROR
echo.
echo 💡 Try these solutions:
echo    1. Close any running applications on ports 8502-8510
echo    2. Run: check_ports.bat to see what's using the ports
echo    3. Kill Python processes: taskkill /f /im python.exe
echo.

:END
echo.
echo 🏁 Application stopped.
echo.
pause
