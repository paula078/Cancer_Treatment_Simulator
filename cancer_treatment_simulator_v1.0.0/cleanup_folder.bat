@echo off
title Cancer Treatment Simulator - Folder Cleanup
color 0E
echo.
echo ================================================================================
echo CANCER TREATMENT SIMULATOR - FOLDER CLEANUP
echo ================================================================================
echo This will remove unnecessary files and keep only what you need
echo ================================================================================
echo.

echo 🧹 Files that will be REMOVED (safe to delete):
echo.
echo 📁 Diagnostic and Testing Files:
echo    - check_completeness.py
echo    - check_ports.bat  
echo    - diagnose_404.py
echo    - diagnose_404_detailed.py
echo    - diagnose_404_detailed.bat
echo    - fix_numba.bat
echo    - fix_permission_error.py
echo    - install_numba_fix.py
echo    - test_streamlit.py
echo    - test_streamlit_direct.bat
echo    - test_streamlit_simple.bat
echo.
echo 📁 Redundant Launchers:
echo    - run_simulator.bat (old version)
echo    - run_simulator_py310.bat (complex version)
echo    - secure_launcher.py (old version)
echo    - secure_launcher_py310.py (complex version)
echo    - launch_direct.bat (redundant)
echo    - launch_firewall_friendly.bat (redundant)
echo    - install_requirements.py (old version)
echo.
echo 📁 Cache and Temporary Files:
echo    - __pycache__/ folder
echo.
echo ✅ Files that will be KEPT (essential):
echo    - main.py (main application)
echo    - treatment_predictor.py (core logic)
echo    - models/ folder (AI models)
echo    - data/ folder (application data)
echo    - .cancer_simulator_env/ (Python environment)
echo    - launch_auto_port.bat (working launcher)
echo    - install_requirements_py310.py (installer)
echo    - cleanup_environment.bat (environment cleanup)
echo    - requirements.txt (dependency list)
echo    - README.md (documentation)
echo.

set /p CONFIRM="Continue with cleanup? This will make your folder much cleaner! (y/N): "
if /i not "%CONFIRM%"=="y" (
    echo Cleanup cancelled.
    pause
    exit /b 0
)

echo.
echo 🗑️  Removing diagnostic files...

REM Remove diagnostic and testing files
if exist "check_completeness.py" del "check_completeness.py" >nul 2>&1
if exist "check_ports.bat" del "check_ports.bat" >nul 2>&1
if exist "diagnose_404.py" del "diagnose_404.py" >nul 2>&1
if exist "diagnose_404_detailed.py" del "diagnose_404_detailed.py" >nul 2>&1
if exist "diagnose_404_detailed.bat" del "diagnose_404_detailed.bat" >nul 2>&1
if exist "fix_numba.bat" del "fix_numba.bat" >nul 2>&1
if exist "fix_permission_error.py" del "fix_permission_error.py" >nul 2>&1
if exist "install_numba_fix.py" del "install_numba_fix.py" >nul 2>&1
if exist "test_streamlit.py" del "test_streamlit.py" >nul 2>&1
if exist "test_streamlit_direct.bat" del "test_streamlit_direct.bat" >nul 2>&1
if exist "test_streamlit_simple.bat" del "test_streamlit_simple.bat" >nul 2>&1

echo ✅ Diagnostic files removed

echo 🗑️  Removing redundant launchers...

REM Remove redundant launchers
if exist "run_simulator.bat" del "run_simulator.bat" >nul 2>&1
if exist "run_simulator_py310.bat" del "run_simulator_py310.bat" >nul 2>&1
if exist "secure_launcher.py" del "secure_launcher.py" >nul 2>&1
if exist "secure_launcher_py310.py" del "secure_launcher_py310.py" >nul 2>&1
if exist "launch_direct.bat" del "launch_direct.bat" >nul 2>&1
if exist "launch_firewall_friendly.bat" del "launch_firewall_friendly.bat" >nul 2>&1
if exist "install_requirements.py" del "install_requirements.py" >nul 2>&1

echo ✅ Redundant launchers removed

echo 🗑️  Removing cache files...

REM Remove cache
if exist "__pycache__" rmdir /s /q "__pycache__" >nul 2>&1

echo ✅ Cache files removed

echo.
echo ================================================================================
echo ✅ CLEANUP COMPLETED SUCCESSFULLY!
echo ================================================================================
echo.
echo 📁 Your folder now contains only essential files:
echo.
echo 🚀 TO RUN THE APPLICATION:
echo    ➤ Double-click: launch_auto_port.bat
echo.
echo 🔧 MAINTENANCE FILES:
echo    ➤ install_requirements_py310.py (reinstall if needed)
echo    ➤ cleanup_environment.bat (clean environment if issues)
echo.
echo 📊 CORE APPLICATION:
echo    ➤ main.py + treatment_predictor.py + models/ + data/
echo    ➤ .cancer_simulator_env/ (Python environment)
echo.
echo Your application is now clean and ready for distribution!
echo.
pause
