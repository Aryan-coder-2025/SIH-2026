@echo off
setlocal
cd /d "%~dp0"
echo ========================================================
echo  SIH26153 Network Attack Forecasting - Evaluator Console
echo ========================================================
echo Launching Streamlit Dashboard from: %~dp0
echo.

python -m streamlit run "%~dp0src\dashboard.py"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Streamlit exited with error code %ERRORLEVEL%.
    pause
)
endlocal
