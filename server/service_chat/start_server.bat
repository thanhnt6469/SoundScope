@echo off
echo ========================================
echo Starting Chat API Server...
echo ========================================
echo.

REM Kill process using port 8003 if exists
echo Checking for existing process on port 8003...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8003 ^| findstr LISTENING') do (
    echo Killing process %%a...
    taskkill /PID %%a /F
    if errorlevel 1 (
        echo Failed to kill process %%a
    ) else (
        echo Successfully killed process %%a
    )
)

REM Wait a moment
echo Waiting for port to be released...
timeout /t 2 /nobreak >nul 2>&1

REM Check if port is free
netstat -ano | findstr :8003 | findstr LISTENING >nul 2>&1
if errorlevel 1 (
    echo Port 8003 is free
) else (
    echo WARNING: Port 8003 is still in use!
    pause
)

REM Start server
echo.
echo ========================================
echo Starting server on http://0.0.0.0:8003...
echo Server will log all requests and uploads
echo Press Ctrl+C to stop the server
echo ========================================
echo.
py -m uvicorn chat_api:app --host 0.0.0.0 --port 8003 --reload --log-level info

