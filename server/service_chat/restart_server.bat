@echo off
echo ========================================
echo RESTARTING CHAT API SERVER
echo ========================================
echo.

echo Step 1: Stopping existing server...
call stop_server.bat

echo.
echo Step 2: Waiting for port to be released...
timeout /t 5 /nobreak >nul 2>&1

echo.
echo Step 3: Verifying port is free...
netstat -ano | findstr :8003 | findstr LISTENING >nul 2>&1
if errorlevel 1 (
    echo   ✅ Port 8003 is free
) else (
    echo   ⚠️  Port 8003 is still in use!
    echo   Force killing remaining processes...
    for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8003 ^| findstr LISTENING') do (
        echo     Killing PID %%a...
        taskkill /PID %%a /F >nul 2>&1
    )
    timeout /t 2 /nobreak >nul 2>&1
)

echo.
echo Step 4: Starting server with new code...
echo   Server will start in this window
echo   Press Ctrl+C to stop
echo.
echo ========================================
echo.

REM Start server
py -m uvicorn chat_api:app --host 0.0.0.0 --port 8003 --reload --log-level info

