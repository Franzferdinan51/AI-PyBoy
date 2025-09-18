@echo off
title AI Game System - Enhanced Auto Startup
setlocal enabledelayedexpansion

REM Enhanced auto startup script with comprehensive error handling and logging
REM Created: 2025-09-18
REM Features: Absolute path handling, detailed logging, port conflict detection, dependency verification

echo ================================================================================
echo                        AI Game System - Enhanced Startup
echo ================================================================================
echo [INFO] Starting enhanced startup script with comprehensive error handling...

REM Create logs directory if it doesn't exist
if not exist "%~dp0logs" mkdir "%~dp0logs"

REM Generate timestamp for log file
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "datetime=%%a"
set "LOGFILE=%~dp0logs\startup_%datetime:~0,4%%datetime:~4,2%%datetime:~6,2%_%datetime:~8,2%%datetime:~10,2%.log"

echo [INFO] Log file: %LOGFILE%
echo [STARTUP] AI Game System Enhanced Startup - %date% %time% > "%LOGFILE%"
echo [STARTUP] Log file: %LOGFILE% >> "%LOGFILE%"
echo [DEBUG] Script directory: %~dp0 >> "%LOGFILE%"
echo [DEBUG] Current working directory: %CD% >> "%LOGFILE%"

REM Function to log and display errors
:log_error
echo [ERROR] %~1
echo [ERROR] %~1 >> "%LOGFILE%"
echo [DEBUG] Error occurred at: %date% %time% >> "%LOGFILE%"
goto :eof

REM Function to log and display warnings
:log_warning
echo [WARNING] %~1
echo [WARNING] %~1 >> "%LOGFILE%"
goto :eof

REM Function to log and display success
:log_success
echo [SUCCESS] %~1
echo [SUCCESS] %~1 >> "%LOGFILE%"
goto :eof

REM Function to log and display info
:log_info
echo [INFO] %~1
echo [INFO] %~1 >> "%LOGFILE%"
goto :eof

REM Function to check if a command exists
:command_exists
where %1 >nul 2>&1
exit /b %errorlevel%

REM ================================================================================
REM Step 1: Directory Structure Verification
REM ================================================================================
echo.
echo [STEP 1/7] Verifying directory structure...
echo [CHECK] Verifying directory structure... >> "%LOGFILE%"

set "BACKEND_PATH=%~dp0ai-game-server\src\main.py"
set "FRONTEND_PATH=%~dp0ai-game-assistant\package.json"

if not exist "%BACKEND_PATH%" (
    call :log_error "Backend main.py not found at: %BACKEND_PATH%"
    echo [DEBUG] Checking backend directory contents... >> "%LOGFILE%"
    if exist "%~dp0ai-game-server" (
        dir "%~dp0ai-game-server" >> "%LOGFILE%" 2>&1
        echo [DEBUG] Backend directory exists, checking src subdirectory... >> "%LOGFILE%"
        if exist "%~dp0ai-game-server\src" (
            dir "%~dp0ai-game-server\src" >> "%LOGFILE%" 2>&1
        ) else (
            echo [DEBUG] src directory does not exist in ai-game-server >> "%LOGFILE%"
        )
    ) else (
        echo [DEBUG] ai-game-server directory does not exist >> "%LOGFILE%"
    )
    echo [SOLUTION] Ensure the backend is properly installed in: %~dp0ai-game-server\src\
    pause
    exit /b 1
)

if not exist "%FRONTEND_PATH%" (
    call :log_error "Frontend package.json not found at: %FRONTEND_PATH%"
    echo [DEBUG] Checking frontend directory contents... >> "%LOGFILE%"
    if exist "%~dp0ai-game-assistant" (
        dir "%~dp0ai-game-assistant" >> "%LOGFILE%" 2>&1
    ) else (
        echo [DEBUG] ai-game-assistant directory does not exist >> "%LOGFILE%"
    )
    echo [SOLUTION] Ensure the frontend is properly installed in: %~dp0ai-game-assistant\
    pause
    exit /b 1
)

call :log_success "Directory structure verified successfully"
echo [OK] Backend: %BACKEND_PATH% >> "%LOGFILE%"
echo [OK] Frontend: %FRONTEND_PATH% >> "%LOGFILE%"

REM ================================================================================
REM Step 2: Python Environment Verification
REM ================================================================================
echo.
echo [STEP 2/7] Verifying Python environment...
echo [CHECK] Verifying Python installation... >> "%LOGFILE%"

call :command_exists python
if %errorlevel% neq 0 (
    call :log_error "Python command not found in PATH"
    echo [SOLUTION] Install Python from python.org and add it to your PATH
    echo [DEBUG] Current PATH: %PATH% >> "%LOGFILE%"
    pause
    exit /b 1
)

REM Get Python version
python --version >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 (
    call :log_error "Python is not working properly"
    echo [SOLUTION] Check Python installation and PATH configuration
    pause
    exit /b 1
)

call :log_success "Python environment verified"
echo [OK] Python version detected >> "%LOGFILE%"

REM ================================================================================
REM Step 3: Node.js Environment Verification
REM ================================================================================
echo.
echo [STEP 3/7] Verifying Node.js environment...
echo [CHECK] Verifying Node.js installation... >> "%LOGFILE%"

call :command_exists node
if %errorlevel% neq 0 (
    call :log_error "Node.js command not found in PATH"
    echo [SOLUTION] Install Node.js from nodejs.org and add it to your PATH
    pause
    exit /b 1
)

REM Get Node.js version
node --version >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 (
    call :log_error "Node.js is not working properly"
    pause
    exit /b 1
)

call :command_exists npm
if %errorlevel% neq 0 (
    call :log_error "npm command not found in PATH"
    echo [SOLUTION] Reinstall Node.js to include npm
    pause
    exit /b 1
)

npm --version >> "%LOGFILE%" 2>&1
call :log_success "Node.js environment verified"
echo [OK] Node.js version detected >> "%LOGFILE%"
echo [OK] npm version detected >> "%LOGFILE%"

REM ================================================================================
REM Step 4: Python Dependencies Setup
REM ================================================================================
echo.
echo [STEP 4/7] Setting up Python dependencies...
echo [BACKEND] Setting up Python dependencies... >> "%LOGFILE%"

pushd "%~dp0ai-game-server"

REM Function to install Python package with error handling
:install_python_package
echo [BACKEND] Checking %~1...
pip show %~1 >> "%LOGFILE%" 2>&1
if %errorlevel% neq 0 (
    echo [BACKEND] Installing %~1...
    pip install %~2 >> "%LOGFILE%" 2>&1
    if %errorlevel% neq 0 (
        if "%~3"=="required" (
            call :log_error "Failed to install required package: %~1"
            popd
            pause
            exit /b 1
        ) else (
            call :log_warning "Failed to install optional package: %~1"
        )
    ) else (
        call :log_success "%~1 installed successfully"
    )
) else (
    echo [BACKEND] %~1 already installed
)
goto :eof

REM Install required packages
call :install_python_package "flask" "flask flask-cors" "required"
call :install_python_package "pillow" "pillow" "required"
call :install_python_package "numpy" "numpy" "required"
call :install_python_package "psutil" "psutil" "required"
call :install_python_package "openai" "openai" "required"

REM Install optional packages
call :install_python_package "pyboy" "pyboy" "optional"
call :install_python_package "pygba" "pygba" "optional"

popd

REM ================================================================================
REM Step 5: Node Dependencies Setup
REM ================================================================================
echo.
echo [STEP 5/7] Setting up Node dependencies...
echo [FRONTEND] Setting up Node dependencies... >> "%LOGFILE%"

pushd "%~dp0ai-game-assistant"

if not exist "node_modules" (
    echo [FRONTEND] Installing dependencies (this may take a while)...
    npm install >> "%LOGFILE%" 2>&1
    if %errorlevel% neq 0 (
        call :log_error "npm install failed"
        echo [DEBUG] npm install output: >> "%LOGFILE%"
        type npm-debug.log >> "%LOGFILE%" 2>&1
        popd
        pause
        exit /b 1
    )
    call :log_success "Frontend dependencies installed successfully"
) else (
    echo [FRONTEND] Dependencies already installed"
)

popd

REM ================================================================================
REM Step 6: Network and Port Verification
REM ================================================================================
echo.
echo [STEP 6/7] Checking network configuration...
echo [NETWORK] Checking for port conflicts... >> "%LOGFILE%"

REM Function to check if port is in use
:check_port
netstat -ano | findstr ":%~1" >> "%LOGFILE%" 2>&1
if %errorlevel% equ 0 (
    call :log_warning "Port %~1 is already in use"
    echo [NETWORK] Process using port %~1: >> "%LOGFILE%"
    netstat -ano | findstr ":%~1" >> "%LOGFILE%"
    echo [SOLUTION] Close the application using port %~1 or let the service use a different port
) else (
    echo [OK] Port %~1 is available"
)
goto :eof

call :check_port "5000"
call :check_port "5173"
call :check_port "5174"
call :check_port "5175"

REM ================================================================================
REM Step 7: Service Startup
REM ================================================================================
echo.
echo [STEP 7/7] Starting services...
echo [SERVICE] Starting backend and frontend services... >> "%LOGFILE%"

REM Start backend service
echo [BACKEND] Starting server on http://localhost:5000...
echo [BACKEND] Executing: cd /d "%~dp0ai-game-server\src" && python main.py >> "%LOGFILE%"
start "AI Game Server" cmd /k "cd /d \"%~dp0ai-game-server\src\" && python main.py"

REM Wait for backend to initialize
echo [WAIT] Waiting for backend to initialize (8 seconds)...
timeout /t 8 >nul

REM Start frontend service
echo [FRONTEND] Starting web UI on http://localhost:5173...
echo [FRONTEND] Executing: cd /d "%~dp0ai-game-assistant\" && npm run dev >> "%LOGFILE%"
start "AI Game Web UI" cmd /k "cd /d \"%~dp0ai-game-assistant\" && npm run dev"

REM Wait a moment for frontend to start
echo [WAIT] Waiting for frontend to initialize (3 seconds)...
timeout /t 3 >nul

REM ================================================================================
REM Final Status and Information
REM ================================================================================
echo.
echo ================================================================================
echo                          Startup Complete!
echo ================================================================================
call :log_success "System startup completed successfully!"

echo.
echo [SYSTEM STATUS]
echo ------------------
echo [✓] Backend Server: http://localhost:5000
echo [✓] Web UI: http://localhost:5173 (or 5174/5175 if 5173 is busy)
echo [✓] Log File: %LOGFILE%
echo.

echo [USAGE INSTRUCTIONS]
echo ---------------------
echo 1. Open your web browser and navigate to the Web UI URL
echo 2. Upload a Game Boy ROM file through the web interface
echo 3. The PyBoy UI will automatically open when you load a ROM
echo 4. Use the AI Assistant to control the game with natural language
echo.

echo [TROUBLESHOOTING]
echo -----------------
echo - If the Web UI shows a different port, use that URL instead
echo - Check the individual command windows for detailed service logs
echo - Review the startup log file: %LOGFILE%
echo - If services fail to start, check for port conflicts
echo - Ensure all dependencies are properly installed
echo.

echo [SYSTEM INFORMATION]
echo --------------------
echo [INFO] Backend: %BACKEND_PATH%
echo [INFO] Frontend: %FRONTEND_PATH%
echo [INFO] Started at: %date% %time%
echo [INFO] Log file: %LOGFILE%
echo [SUCCESS] System startup completed! >> "%LOGFILE%"
echo [INFO] Log file location saved for reference >> "%LOGFILE%"

echo.
echo Press any key to close this window (services will continue running)...
pause >nul

endlocal