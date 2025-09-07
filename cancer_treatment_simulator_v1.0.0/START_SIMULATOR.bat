@echo off
title Cancer Treatment Simulator - Auto Port Launcher
color 0A
echo.
echo ================================================================================
echo CANCER TREATMENT SIMULATOR - AUTO PORT LAUNCHER
echo ================================================================================
echo Automatically finding free port to avoid conflicts
echo ================================================================================
echo.

REM Navigate to the script directory
cd /d "%~dp0"

echo 🔍 Finding available port...

REM Use virtual environment if available
if exist ".cancer_simulator_env\Scripts\python.exe" (
    echo ✅ Using virtual environment Python...
    set PYTHON_CMD=".cancer_simulator_env\Scripts\python.exe"
) else (
    echo ⚠️  No virtual environment found, using system Python...
    set PYTHON_CMD=py -3.10
)

REM Try different ports starting from 8502 (avoid 8501 which might be blocked)
for %%p in (8502 8503 8504 8505 8506 8507 8508 8509 8510) do (
    echo 🔍 Trying port %%p...
    
    REM Check if port is free using netstat
    netstat -an | find "%%p" >nul
    if errorlevel 1 (
        echo ✅ Port %%p is available!
        echo.
        echo 🚀 Starting Cancer Treatment Simulator on http://localhost:%%p
        echo.
        echo 💡 Your application will open automatically in your browser
        echo 💡 Close this window to stop the application
        echo.
        
        REM Start Streamlit on the free port
        %PYTHON_CMD% -m streamlit run main.py --server.port %%p --server.address localhost
        goto END
    ) else (
        echo ⚠️  Port %%p is busy, trying next...
    )
)

echo.
echo ❌ All common ports (8502-8510) are busy!
echo.
echo 💡 Try these solutions:
echo    1. Close any running Streamlit applications
echo    2. Close browsers and try again
echo    3. Restart your computer
echo    4. Check Windows Firewall settings
echo.

:END
echo.
echo 🏁 Application stopped.
echo.
pause
