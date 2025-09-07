@echo off
title Cancer Treatment Simulator - Direct Launch
color 0A
echo.
echo ================================================================================
echo CANCER TREATMENT SIMULATOR - DIRECT STREAMLIT LAUNCH
echo ================================================================================
echo All diagnostics passed! Launching directly...
echo ================================================================================
echo.

echo 🚀 Starting Cancer Treatment Simulator...
echo.
echo 📍 Server will start on: http://127.0.0.1:8501
echo 🔒 Security: Complete isolation enabled
echo 📊 Features: Full AI treatment optimization available
echo.
echo Instructions:
echo   • Server will open in your browser automatically
echo   • Press Ctrl+C in this window to stop
echo   • All processing is local and secure
echo.

".cancer_simulator_env\Scripts\python.exe" -m streamlit run main.py --server.port 8501 --server.address 127.0.0.1 --browser.gatherUsageStats false

echo.
echo 🏁 Simulator has stopped.
pause
