@echo off
title Cancer Treatment Simulator - Environment Cleanup
color 0C
echo.
echo ================================================================================
echo CANCER TREATMENT SIMULATOR - ENVIRONMENT CLEANUP UTILITY
echo ================================================================================
echo This utility will clean up the virtual environment to resolve permission issues
echo ================================================================================
echo.

echo 🧹 Checking for existing virtual environment...

if not exist ".cancer_simulator_env" (
    echo ✅ No virtual environment found - cleanup not needed
    echo You can now run: run_simulator_py310.bat
    pause
    exit /b 0
)

echo ⚠️  Found existing virtual environment
echo.
echo This will completely remove the current virtual environment.
echo A new Python 3.10 environment will be created when you run the simulator.
echo.
set /p CONFIRM="Continue with cleanup? (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Cleanup cancelled.
    pause
    exit /b 0
)

echo.
echo 🔄 Stopping any running processes...
taskkill /f /im python.exe >nul 2>&1
taskkill /f /im pythonw.exe >nul 2>&1

echo 🗑️  Removing virtual environment...
echo This may take a moment...

REM First try normal removal
rmdir /s /q ".cancer_simulator_env" >nul 2>&1

REM Check if removal was successful
if not exist ".cancer_simulator_env" (
    echo ✅ Virtual environment removed successfully
    goto SUCCESS
)

echo ⏳ Some files are locked, trying alternative method...

REM Try with takeown and icacls to unlock files
takeown /f ".cancer_simulator_env" /r /d y >nul 2>&1
icacls ".cancer_simulator_env" /grant administrators:F /t >nul 2>&1
rmdir /s /q ".cancer_simulator_env" >nul 2>&1

REM Check again
if not exist ".cancer_simulator_env" (
    echo ✅ Virtual environment removed successfully (method 2)
    goto SUCCESS
)

echo ⏳ Still locked, trying force delete...

REM Use robocopy to create empty folder and merge (this often works)
mkdir temp_empty_folder >nul 2>&1
robocopy temp_empty_folder ".cancer_simulator_env" /mir >nul 2>&1
rmdir /s /q ".cancer_simulator_env" >nul 2>&1
rmdir /s /q temp_empty_folder >nul 2>&1

REM Final check
if not exist ".cancer_simulator_env" (
    echo ✅ Virtual environment removed successfully (method 3)
    goto SUCCESS
)

REM If all else fails, provide manual instructions
echo.
echo ❌ Automatic cleanup failed
echo.
echo Manual cleanup required:
echo 1. Close all Python processes and VS Code instances
echo 2. Restart your computer
echo 3. Manually delete the .cancer_simulator_env folder
echo 4. Run run_simulator_py310.bat again
echo.
echo Or try running this cleanup utility again after restart.
pause
exit /b 1

:SUCCESS
echo.
echo ================================================================================
echo ✅ CLEANUP COMPLETED SUCCESSFULLY!
echo ================================================================================
echo.
echo 🚀 Ready to create fresh Python 3.10 environment
echo.
echo Next steps:
echo   1. Double-click: run_simulator_py310.bat
echo   2. Or run: py -3.10 install_requirements_py310.py
echo.
echo The new environment will be optimized for Python 3.10!
echo.
pause
