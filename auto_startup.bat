@echo off
title AI Game System - Auto Startup (Smart Dependencies)
echo [STARTUP] Launching AI Game System...

REM Basic checks
if not exist "ai-game-server\src\main.py" (
    echo [ERROR] ai-game-server\src\main.py not found!
    pause
    exit /b 1
)
if not exist "ai-game-assistant\package.json" (
    echo [ERROR] ai-game-assistant\package.json not found!
    pause
    exit /b 1
)
echo [OK] Directories OK

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from python.org
    pause
    exit /b 1
)
echo [OK] Python OK

REM Check Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Install from nodejs.org
    pause
    exit /b 1
)
echo [OK] Node.js OK

REM Install Python dependencies only if not present
echo [BACKEND DEP] Checking Python dependencies...
pushd "%~dp0ai-game-server"
pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Flask and dependencies...
    pip install flask flask-cors pillow numpy psutil openai
) else (
    echo Flask already installed
)
pip show pyboy >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing PyBoy...
    pip install pyboy
) else (
    echo PyBoy already installed
)

echo [INFO] Checking for PyGBA...
pip show pygba
if %errorlevel% neq 0 (
    echo [INFO] PyGBA not found. Attempting installation...
    pip install pygba
    if %errorlevel% neq 0 (
        echo [WARNING] PyGBA installation failed. GBA support will be disabled.
    ) else (
        echo [SUCCESS] PyGBA installed successfully.
    )
) else (
    echo [INFO] PyGBA is already installed.
)
popd

REM Install Node dependencies only if node_modules doesn't exist
echo [FRONTEND DEP] Checking Node dependencies...
pushd "%~dp0ai-game-assistant"
if not exist "node_modules" (
    echo [INFO] 'node_modules' not found. Running 'npm install'...
    npm install
) else (
    echo [INFO] 'node_modules' already exists. Skipping 'npm install'.
)
popd

REM Start backend
echo [BACKEND] Starting server on http://localhost:5000 (PyBoy UI opens on ROM load)...
start "AI Game Server" cmd /k "cd /d %~dp0ai-game-server\src && python main.py"

REM Wait 5 seconds for backend to start
timeout /t 5 >nul

REM Start frontend
echo [FRONTEND] Starting web UI on http://localhost:5173...
start "AI Game Web UI" cmd /k "cd /d %~dp0ai-game-assistant && npm run dev"

echo [SUCCESS] System started!
echo Backend API: http://localhost:5000 (check window for Flask logs)
echo Web UI: http://localhost:5173 (open in browser)
echo [INFO] Upload GB ROM in web UI to open PyBoy UI. PyGBA warning is normal if not installed.
pause