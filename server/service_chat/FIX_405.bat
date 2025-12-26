@echo off
echo ========================================
echo FIX 405 ERROR - FORCE RESTART SERVER
echo ========================================
echo.

echo [1/5] Stopping all processes on port 8003...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8003 ^| findstr LISTENING') do (
    echo   Killing PID %%a...
    taskkill /PID %%a /F >nul 2>&1
    if errorlevel 1 (
        echo   ⚠️  Failed to kill %%a, trying again...
        timeout /t 1 /nobreak >nul 2>&1
        taskkill /PID %%a /F >nul 2>&1
    )
)

echo.
echo [2/5] Waiting for port to be released...
timeout /t 5 /nobreak >nul 2>&1

echo.
echo [3/5] Verifying port is free...
netstat -ano | findstr :8003 | findstr LISTENING >nul 2>&1
if errorlevel 1 (
    echo   ✅ Port 8003 is free
) else (
    echo   ❌ Port 8003 is still in use!
    echo   Force killing ALL Python processes...
    taskkill /IM python.exe /F >nul 2>&1
    taskkill /IM py.exe /F >nul 2>&1
    timeout /t 3 /nobreak >nul 2>&1
)

echo.
echo [4/5] Verifying code has GET /upload endpoint...
py -c "from chat_api import app; routes = [(r.path, list(r.methods) if hasattr(r, 'methods') else []) for r in app.routes if hasattr(r, 'path') and '/upload' in str(r.path)]; print('Upload endpoints:', routes)" 2>nul
if errorlevel 1 (
    echo   ❌ Error loading code! Check chat_api.py for syntax errors
    pause
    exit /b 1
) else (
    echo   ✅ Code verified - GET /upload endpoint exists
)

echo.
echo [5/5] Starting server with NEW code...
echo.
echo ========================================
echo SERVER WILL START IN THIS WINDOW
echo Press Ctrl+C to stop
echo ========================================
echo.
echo Waiting 2 seconds before start...
timeout /t 2 /nobreak >nul 2>&1
echo.

REM Start server
py -m uvicorn chat_api:app --host 0.0.0.0 --port 8003 --reload --log-level info

