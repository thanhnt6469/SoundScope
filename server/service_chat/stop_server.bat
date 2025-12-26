@echo off
echo Stopping Chat API Server...
echo.

REM Find and kill process using port 8003
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8003 ^| findstr LISTENING') do (
    echo Found process %%a on port 8003
    taskkill /PID %%a /F
    if errorlevel 1 (
        echo Failed to kill process %%a
    ) else (
        echo Successfully stopped process %%a
    )
)

echo.
echo Done!

