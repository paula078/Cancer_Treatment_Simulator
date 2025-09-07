@echo off
title Cancer Treatment Simulator - Port Diagnostic
color 0E
echo.
echo ================================================================================
echo CANCER TREATMENT SIMULATOR - PORT DIAGNOSTIC
echo ================================================================================
echo Checking what's using common Streamlit ports
echo ================================================================================
echo.

echo 🔍 Checking port usage...
echo.

echo 📊 Ports 8501-8510 status:
echo.
for %%p in (8501 8502 8503 8504 8505 8506 8507 8508 8509 8510) do (
    echo Checking port %%p:
    netstat -ano | find ":%%p " >nul
    if errorlevel 1 (
        echo   ✅ Port %%p: FREE
    ) else (
        echo   ❌ Port %%p: BUSY
        echo      Details:
        netstat -ano | find ":%%p " | head -3
    )
    echo.
)

echo.
echo 🔍 Looking for any Streamlit processes...
echo.
tasklist | find /i "python" >nul
if not errorlevel 1 (
    echo 📋 Python processes found:
    tasklist | find /i "python"
    echo.
    echo 💡 If you see multiple Python processes, some might be Streamlit
    echo    You can kill them with: taskkill /f /im python.exe
) else (
    echo ✅ No Python processes found
)

echo.
echo 🔍 Checking Windows Firewall status...
echo.
netsh advfirewall show currentprofile | find "State"

echo.
echo ================================================================================
echo 📝 RECOMMENDATIONS:
echo ================================================================================
echo.
echo If ports are busy:
echo   1. Close any running Streamlit applications
echo   2. Kill Python processes: taskkill /f /im python.exe
echo   3. Try the auto-port launcher: launch_auto_port.bat
echo.
echo If Windows Firewall is blocking:
echo   1. Try running as Administrator
echo   2. Add Python to firewall exceptions
echo   3. Use 127.0.0.1 instead of localhost
echo.
pause
