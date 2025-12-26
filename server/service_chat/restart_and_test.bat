@echo off
echo ========================================
echo Restarting Server and Testing
echo ========================================
echo.

REM Kill existing processes
echo Step 1: Stopping existing server...
call stop_server.bat

REM Wait
timeout /t 2 /nobreak >nul 2>&1

REM Start server in background
echo Step 2: Starting server...
start /B py -m uvicorn chat_api:app --host 0.0.0.0 --port 8003 --reload --log-level info

REM Wait for server to start
echo Step 3: Waiting for server to start...
timeout /t 5 /nobreak >nul 2>&1

REM Test endpoints
echo Step 4: Testing endpoints...
echo.

echo Testing GET /upload...
py -c "import requests; import time; time.sleep(1); r = requests.get('http://localhost:8003/upload'); print(f'Status: {r.status_code}'); print(f'Response: {r.json() if r.status_code == 200 else r.text}')"

echo.
echo Testing POST /upload...
py -c "import requests; files = {'file': ('test.txt', b'Test content', 'text/plain')}; r = requests.post('http://localhost:8003/upload', files=files); print(f'Status: {r.status_code}'); print(f'Response: {r.json() if r.status_code == 200 else r.text}')"

echo.
echo ========================================
echo Test completed!
echo Server is running in background
echo Check the server window for logs
echo ========================================
pause

