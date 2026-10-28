@echo off
title Springer 99.88% Real-Time Indian Sign Language Engine
echo ==============================================================
echo  Launching Springer ISL Real-Time Recognition System
echo ==============================================================
echo [1] CNN+LSTM Model (17 Core Words - 99.88% Benchmark Accuracy)
echo [2] StackLSTM Model (61 Full ISLRTC Dynamic ISL Words)
echo.
echo Controls inside Camera Window:
echo   - Press '1': Switch to CNN+LSTM (17 signs)
echo   - Press '2': Switch to StackLSTM (61 signs)
echo   - Press 'q': Exit camera
echo ==============================================================
echo.
"C:\Users\Asus\AppData\Local\Programs\Python\Python312\python.exe" "%~dp0research_model\run_live_isl.py"
pause
