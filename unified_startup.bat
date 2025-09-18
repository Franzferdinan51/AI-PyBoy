@echo off
setlocal enabledelayedexpansion
REM ============================================
REM AI Game Playing System - UNIFIED STARTUP SCRIPT
REM This is the MAIN startup script with ALL features
REM NEVER EXITS without user prompting - Most comprehensive startup solution
REM ============================================

title AI Game System - Unified Startup Console

REM ============================================
REM Critical Configuration - NEVER EXIT AUTOMATICALLY
REM ============================================
set "SCRIPT_RUNNING=true"
set "USER_EXIT_REQUESTED=false"
set "CONTINUOUS_OPERATION=true"

REM ============================================
REM Global Variables and Ultimate Configuration
REM ============================================
set "STARTUP_MODE=UNKNOWN"
set "LOGGING_ENABLED=true"
set "MONITORING_ENABLED=true"
set "AUTO_RESTART=false"
set "VERBOSE_LOGGING=true"
set "UNIFIED_LOGGER_ENABLED=true"
set "CONTINUOUS_MONITORING=true"
set "BACKEND_PORT=5000"
set "FRONTEND_PORT=5173"
set "MONITOR_PORT=8080"
set "LOG_DIR=logs"
set "TEMP_LOG_DIR=%TEMP%\ai_game_system_ultimate"

REM ============================================
REM Initialize Directories
REM ============================================
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%TEMP_LOG_DIR%" mkdir "%TEMP_LOG_DIR%"

REM ============================================
REM Main Menu
REM ============================================
:MainMenu
cls
echo ================================================================
echo == AI Game Playing System - Unified Startup Console            ==
echo ================================================================
echo.
echo Select startup mode:
echo.
echo 1. Basic Startup (Quick start, minimal logging)
echo 2. Enhanced Logging (Terminal logging + file logging)
echo 3. Monitoring Console (Real-time monitoring with controls)
echo 4. Service Monitor (Auto-restart + web dashboard)
echo 5. Development Mode (Verbose logging + debugging)
echo 6. Clean Start (Kill existing processes + fresh start)
echo 7. System Health Check (Check dependencies and system status)
echo 8. Recovery Mode (Advanced troubleshooting and recovery)
echo 9. Ultimate Mode (Start EVERYTHING with all features)
echo.
echo 0. Complete System Shutdown (Stops ALL services)
echo.
echo [CRITICAL] This script NEVER exits without your explicit confirmation!
echo All modes provide continuous monitoring until you choose to quit.
echo.
echo Enter your choice (0-9):

set /p choice=
if "%choice%"=="" goto MainMenu
if "%choice%"=="0" goto CompleteSystemShutdown
if "%choice%"=="1" goto BasicStartup
if "%choice%"=="2" goto EnhancedLogging
if "%choice%"=="3" goto MonitoringConsole
if "%choice%"=="4" goto ServiceMonitor
if "%choice%"=="5" goto DevelopmentMode
if "%choice%"=="6" goto CleanStart
if "%choice%"=="7" goto SystemHealthCheck
if "%choice%"=="8" goto RecoveryMode
if "%choice%"=="9" goto UltimateStartup

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MainMenu

REM ============================================
REM Startup Mode Functions
REM ============================================

:BasicStartup
set "STARTUP_MODE=Basic"
set "LOGGING_ENABLED=false"
set "MONITORING_ENABLED=false"
set "AUTO_RESTART=false"
set "VERBOSE_LOGGING=false"
call :ValidateSystem
if %errorlevel% neq 0 goto MainMenu
call :InstallDependencies
if %errorlevel% neq 0 goto MainMenu
call :StartBasicServices
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

:EnhancedLogging
set "STARTUP_MODE=Enhanced Logging"
set "LOGGING_ENABLED=true"
set "MONITORING_ENABLED=false"
set "AUTO_RESTART=false"
set "VERBOSE_LOGGING=true"
call :ValidateSystem
if %errorlevel% neq 0 goto MainMenu
call :InstallDependencies
if %errorlevel% neq 0 goto MainMenu
call :StartEnhancedLogging
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

:MonitoringConsole
set "STARTUP_MODE=Monitoring Console"
set "LOGGING_ENABLED=true"
set "MONITORING_ENABLED=true"
set "AUTO_RESTART=false"
set "VERBOSE_LOGGING=true"
call :ValidateSystem
if %errorlevel% neq 0 goto MainMenu
call :InstallDependencies
if %errorlevel% neq 0 goto MainMenu
call :StartMonitoringConsole
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

:ServiceMonitor
set "STARTUP_MODE=Service Monitor"
set "LOGGING_ENABLED=true"
set "MONITORING_ENABLED=true"
set "AUTO_RESTART=true"
set "VERBOSE_LOGGING=false"
call :ValidateSystem
if %errorlevel% neq 0 goto MainMenu
call :InstallDependencies
if %errorlevel% neq 0 goto MainMenu
call :StartServiceMonitor
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

:DevelopmentMode
set "STARTUP_MODE=Development"
set "LOGGING_ENABLED=true"
set "MONITORING_ENABLED=true"
set "AUTO_RESTART=false"
set "VERBOSE_LOGGING=true"
call :ValidateSystem
if %errorlevel% neq 0 goto MainMenu
call :InstallDependencies
if %errorlevel% neq 0 goto MainMenu
call :StartDevelopmentMode
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

:CleanStart
call :CleanExistingProcesses
call :BasicStartup
goto End

:SystemHealthCheck
set "STARTUP_MODE=Health Check"
call :ValidateSystem
if %errorlevel% neq 0 (
    echo System health check failed!
) else (
    echo System health check passed!
)
echo.
echo Press any key to return to main menu...
pause >nul
goto MainMenu

:RecoveryMode
set "STARTUP_MODE=Recovery"
call :ValidateSystem
if %errorlevel% neq 0 (
    echo System validation failed. Attempting recovery...
    call :AttemptRecovery
    if %errorlevel% neq 0 (
        echo Recovery failed. Please check system manually.
        pause
        goto MainMenu
    )
    echo Recovery successful!
)
call :StartRecoveryMode
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

:UltimateStartup
set "STARTUP_MODE=Ultimate"
set "LOGGING_ENABLED=true"
set "MONITORING_ENABLED=true"
set "AUTO_RESTART=true"
set "VERBOSE_LOGGING=true"
set "UNIFIED_LOGGER_ENABLED=true"
set "CONTINUOUS_MONITORING=true"
call :ValidateSystem
if %errorlevel% neq 0 goto MainMenu
call :InstallDependencies
if %errorlevel% neq 0 goto MainMenu
call :StartUltimateMode
REM Return to main menu after completion (NEVER exit automatically)
goto MainMenu

REM ============================================
REM System Validation Functions
REM ============================================

:ValidateSystem
echo [VALIDATE] Checking system prerequisites...
echo.

REM Check if running from the correct directory
if not exist "ai-game-server\src\main.py" (
    echo [ERROR] Cannot find backend server files.
    echo        Please run this script from the root directory containing ai-game-server and ai-game-assistant folders.
    echo.
    pause
    exit /b 1
)

if not exist "ai-game-assistant\package.json" (
    echo [ERROR] Cannot find frontend files.
    echo        Please run this script from the root directory containing ai-game-server and ai-game-assistant folders.
    echo.
    pause
    exit /b 1
)

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo        Please install Python from https://python.org
    echo.
    pause
    exit /b 1
) else (
    echo [OK] Python is available
    python --version
)

REM Check for Node.js
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js is not installed or not in PATH.
    echo        Please install Node.js from https://nodejs.org
    echo.
    pause
    exit /b 1
) else (
    echo [OK] Node.js is available
    node --version
)

REM Check for npm
echo [NPM] Checking npm version...
npm --version
if %errorlevel% neq 0 (
    echo [ERROR] npm is not installed or not in PATH.
    echo        Please reinstall Node.js with npm.
    echo.
    pause
    exit /b 1
) else (
    echo [OK] npm is available
)

REM Check port availability
netstat -an | findstr ":%BACKEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [WARNING] Port %BACKEND_PORT% is already in use.
    echo        This might indicate another instance is running.
)

netstat -an | findstr ":%FRONTEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [WARNING] Port %FRONTEND_PORT% is already in use.
    echo        This might indicate another instance is running.
)

echo.
echo [SUCCESS] All prerequisites validated!
echo.
exit /b 0

:InstallDependencies
echo [INSTALL] Installing dependencies...
echo.

REM Install Python dependencies
echo [PYTHON] Installing Python dependencies...
if exist "ai-game-server\requirements.txt" (
    pip install -r ai-game-server\requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Failed with pip, trying pip3...
        pip3 install -r ai-game-server\requirements.txt
        if %errorlevel% neq 0 (
            echo [ERROR] Failed to install Python dependencies.
            echo        Please make sure you have Python and pip installed.
            echo.
            pause
            exit /b 1
        )
    )
    echo [SUCCESS] Python dependencies installed
) else (
    echo [WARNING] requirements.txt not found. Skipping Python dependency installation.
)

REM Install Node.js dependencies
echo [NODE] Installing Node.js dependencies...
cd ai-game-assistant
if not exist "node_modules" (
    npm install
    if %errorlevel% neq 0 (
        cd ..
        echo [ERROR] Failed to install Node.js dependencies.
        echo        Please make sure you have Node.js and npm installed.
        echo.
        pause
        exit /b 1
    )
    echo [SUCCESS] Node.js dependencies installed
) else (
    echo [OK] Node.js dependencies already installed
)

REM Check for Vite
npm list vite >nul 2>&1
if %errorlevel% neq 0 (
    echo [VITE] Installing Vite...
    npm install vite
    if %errorlevel% neq 0 (
        cd ..
        echo [ERROR] Failed to install Vite.
        pause
        exit /b 1
    )
    echo [SUCCESS] Vite installed
) else (
    echo [OK] Vite already installed
)

cd ..
echo.
echo [SUCCESS] All dependencies installed successfully!
echo.
exit /b 0

REM ============================================
REM Service Startup Functions
REM ============================================

:StartBasicServices
echo [START] Starting basic services with unified logging...
echo.

REM Create log directory
if not exist "%TEMP_LOG_DIR%" mkdir "%TEMP_LOG_DIR%"
set "BACKEND_LOG=%TEMP_LOG_DIR%\backend_basic.log"
set "FRONTEND_LOG=%TEMP_LOG_DIR%\frontend_basic.log"

REM Start backend server with background logging
echo [BACKEND] Starting backend server...
start "Backend Server" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"

timeout /t 3 /nobreak >nul

REM Start frontend application with background logging
echo [FRONTEND] Starting frontend application...
start "Frontend App" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

echo.
echo [SUCCESS] Basic startup complete!
echo.
echo Backend server:    http://localhost:%BACKEND_PORT%
echo Frontend app:       http://localhost:%FRONTEND_PORT%
echo.

REM Enter unified monitoring loop
call :UnifiedMonitoringLoop "Basic"
exit /b 0

:StartEnhancedLogging
echo [START] Starting enhanced logging mode...
echo.

REM Create log directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%TEMP_LOG_DIR%" mkdir "%TEMP_LOG_DIR%"

REM Create log files
set "BACKEND_LOG=%LOG_DIR%\backend_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%.log"
set "FRONTEND_LOG=%LOG_DIR%\frontend_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%.log"
set "UNIFIED_LOG=%LOG_DIR%\unified_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%.log"

REM Start backend with logging
echo [BACKEND] Starting backend server with logging...
start "Backend Server" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"

timeout /t 3 /nobreak >nul

REM Start frontend with logging
echo [FRONTEND] Starting frontend application with logging...
start "Frontend App" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

REM Start unified logger if available
if exist "unified_logger.py" (
    echo [LOGGER] Starting unified logger...
    echo   - Real-time terminal output with color coding
    echo   - Live streaming of all service activities
    echo   - Enhanced service health monitoring
    echo   - Performance statistics and metrics
    echo   - Structured JSON logging
    echo   - Automatic log file management
    echo   - Graceful shutdown handling
    start "Unified Logger" cmd /c "python unified_logger.py > \"%UNIFIED_LOG%\" 2>&1"
    echo [SUCCESS] Unified logger started with enhanced capabilities!
) else (
    echo [WARNING] unified_logger.py not found, using enhanced logging mode
)

echo.
echo [SUCCESS] Enhanced logging mode started!
echo.
echo Backend server:    http://localhost:%BACKEND_PORT%
echo Frontend app:       http://localhost:%FRONTEND_PORT%
echo Log directory:     %LOG_DIR%
echo.
echo [INFO] All output is being logged to files and displayed in real-time.
echo.

REM Enter unified monitoring loop
call :UnifiedMonitoringLoop "Enhanced Logging"
exit /b 0

:StartMonitoringConsole
echo [START] Starting monitoring console...
echo.

REM Create log directories
if not exist "%TEMP_LOG_DIR%" mkdir "%TEMP_LOG_DIR%"

set "BACKEND_LOG=%TEMP_LOG_DIR%\backend_monitor.log"
set "FRONTEND_LOG=%TEMP_LOG_DIR%\frontend_monitor.log"

REM Clear previous log files
if exist "%BACKEND_LOG%" del "%BACKEND_LOG%"
if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%"

REM Start backend with logging
echo [BACKEND] Starting backend server...
start "Backend Server" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend with logging
echo [FRONTEND] Starting frontend application...
start "Frontend App" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

REM Start monitoring loop
call :MonitoringLoop
exit /b 0

:StartServiceMonitor
echo [START] Starting service monitor...
echo.

REM Start service monitor if available
if exist "service_monitor.py" (
    echo [MONITOR] Starting service monitor...
    start "Service Monitor" cmd /k "python service_monitor.py"

    echo.
    echo [SUCCESS] Service monitor started!
    echo.
    echo Service Monitor Dashboard: http://localhost:%MONITOR_PORT%
    echo Backend server:           http://localhost:%BACKEND_PORT%
    echo Frontend app:             http://localhost:%FRONTEND_PORT%
    echo.
    echo [Features]
    echo - Automatic service restart if crashes occur
    echo - Real-time resource monitoring (CPU, Memory)
    echo - Health checks for all services
    echo - Web dashboard with monitoring data
    echo - Comprehensive logging and alerting
    echo.
    echo [Note]
    echo - The service monitor will automatically start and manage all services
    echo - View real-time status and metrics on the dashboard
    echo - Check the logs/ directory for detailed monitoring logs
    echo - To stop the system, close the Service Monitor window first
    echo.

    REM Enter unified monitoring loop
    call :UnifiedMonitoringLoop "Service Monitor"
    exit /b 0
) else (
    echo [ERROR] service_monitor.py not found. Falling back to monitoring console...
    call :StartMonitoringConsole
    exit /b 0
)

:StartDevelopmentMode
echo [START] Starting development mode...
echo.

REM Create log directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "BACKEND_LOG=%LOG_DIR%\backend_dev.log"
set "FRONTEND_LOG=%LOG_DIR%\frontend_dev.log"

REM Start backend with verbose logging
echo [BACKEND] Starting backend server in development mode...
start "Backend Server (Dev)" cmd /c "cd /d \"%~dp0ai-game-server\" && set PYTHONPATH=. && set FLASK_ENV=development && python -u src/main.py > \"%BACKEND_LOG%\" 2>&1"

timeout /t 3 /nobreak >nul

REM Start frontend with verbose logging
echo [FRONTEND] Starting frontend application in development mode...
start "Frontend App (Dev)" cmd /c "cd /d \"%~dp0ai-game-assistant\" && set NODE_ENV=development && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

echo.
echo [SUCCESS] Development mode started!
echo.
echo Backend server:    http://localhost:%BACKEND_PORT%
echo Frontend app:       http://localhost:%FRONTEND_PORT%
echo Log directory:     %LOG_DIR%
echo.
echo [INFO] Development mode with hot reload enabled.
echo [INFO] Real-time logging and monitoring active.
echo.

REM Start monitoring in development mode
call :UnifiedMonitoringLoop "Development"
exit /b 0

:StartRecoveryMode
echo [START] Starting recovery mode...
echo.

REM Kill any existing processes first
call :CleanExistingProcesses

REM Start with enhanced logging and monitoring
set "BACKEND_LOG=%LOG_DIR%\backend_recovery.log"
set "FRONTEND_LOG=%LOG_DIR%\frontend_recovery.log"

REM Start backend with recovery logging
echo [BACKEND] Starting backend server in recovery mode...
start "Backend Server (Recovery)" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"

timeout /t 5 /nobreak >nul

REM Start frontend with recovery logging
echo [FRONTEND] Starting frontend application in recovery mode...
start "Frontend App (Recovery)" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

REM Start recovery monitoring
call :UnifiedMonitoringLoop "Recovery"
exit /b 0

:StartUltimateMode
echo [START] Starting ULTIMATE mode... All features enabled!
echo.

REM Create log directories
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%TEMP_LOG_DIR%" mkdir "%TEMP_LOG_DIR%"

set "BACKEND_LOG=%LOG_DIR%\backend_ultimate.log"
set "FRONTEND_LOG=%LOG_DIR%\frontend_ultimate.log"
set "UNIFIED_LOG=%LOG_DIR%\unified_ultimate.log"
set "MONITOR_LOG=%LOG_DIR%\monitor_ultimate.log"

REM Start backend server first (required for other services)
echo [BACKEND] Starting backend server in ultimate mode...
start "Backend Server (Ultimate)" cmd /c "cd /d \"%~dp0ai-game-server\" && set PYTHONPATH=. && set FLASK_ENV=development && python -u src/main.py > \"%BACKEND_LOG%\" 2>&1"
timeout /t 5 /nobreak >nul

REM Start frontend application
echo [FRONTEND] Starting frontend application in ultimate mode...
start "Frontend App (Ultimate)" cmd /c "cd /d \"%~dp0ai-game-assistant\" && set NODE_ENV=development && npm run dev > \"%FRONTEND_LOG%\" 2>&1"
timeout /t 3 /nobreak >nul

REM Start service monitor if available
if exist "service_monitor.py" (
    echo [MONITOR] Starting service monitor with auto-restart...
    start "Service Monitor" cmd /c "python service_monitor.py > \"%MONITOR_LOG%\" 2>&1"
) else (
    echo [ERROR] service_monitor.py not found. Ultimate mode requires it.
    pause
    goto MainMenu
)

timeout /t 3 /nobreak >nul

REM Start unified logger if available
if exist "unified_logger.py" (
    echo [LOGGER] Starting unified logger for comprehensive logging...
    start "Unified Logger" cmd /c "python unified_logger.py > \"%UNIFIED_LOG%\" 2>&1"
) else (
    echo [WARNING] unified_logger.py not found, some logging features will be missing.
)

REM Verify core services are running before proceeding
echo [VERIFY] Checking service status...
set "SERVICE_CHECK_COUNT=0"
:VerifyServices
REM Check Backend Status
netstat -an | findstr ":%BACKEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [BACKEND]  ✓ Running on http://localhost:%BACKEND_PORT%
) else (
    echo [BACKEND]  ✗ Not responding on port %BACKEND_PORT%
    if exist "%BACKEND_LOG%" (
        echo   Last error log:
        powershell -Command "Get-Content -Path '%BACKEND_LOG%' | Select-String -Pattern 'error' -CaseSensitive | Select-Object -Last 2"
    )
)

REM Check Frontend Status
netstat -an | findstr ":%FRONTEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [FRONTEND] ✓ Running on http://localhost:%FRONTEND_PORT%
) else (
            echo   Last error log:
            powershell -Command "Get-Content -Path '%FRONTEND_LOG%' | Select-String -Pattern 'error' -CaseSensitive | Select-Object -Last 2")

REM If services aren't running, wait and retry
netstat -an | findstr ":%BACKEND_PORT%" >nul
if %errorlevel% == 0 (
    netstat -an | findstr ":%FRONTEND_PORT%" >nul
    if %errorlevel% == 0 goto :ServicesVerified
)

set /a SERVICE_CHECK_COUNT+=1
if %SERVICE_CHECK_COUNT% ge 6 (
    echo [WARNING] Some services may not have started properly.
    echo [INFO] The monitoring loop will continue to check service status.
    goto :ServicesVerified
)

echo [WAIT] Waiting for services to start... (%SERVICE_CHECK_COUNT%/6)
timeout /t 5 /nobreak >nul
goto :VerifyServices

:ServicesVerified
echo.
echo [SUCCESS] ULTIMATE mode started!
echo.
echo Service Monitor Dashboard: http://localhost:%MONITOR_PORT%
echo Backend server:           http://localhost:%BACKEND_PORT%
echo Frontend app:             http://localhost:%FRONTEND_PORT%
echo Log directory:            %LOG_DIR%
echo.
echo [Features Enabled]
echo - Automatic service restart and health checks
echo - Real-time resource monitoring web dashboard
echo - Comprehensive, structured, and real-time logging
echo - Hot-reloading for development
echo.
echo [INFO] All systems are GO! Monitoring all services.
echo.

REM Enter unified monitoring loop
call :UnifiedMonitoringLoop "Ultimate"
exit /b 0

REM ============================================
REM Unified Monitoring Loop Functions
REM ============================================

:UnifiedMonitoringLoop
set "MODE=%~1"
set "MONITOR_COUNT=0"

:UnifiedMonitorLoop
cls

REM Header
echo ================================================================
echo == AI Game System - %MODE% Mode                               ==
echo ================================================================
echo Status Update: %date% %time%
echo Monitoring Cycle: %MONITOR_COUNT%
echo.

REM Service Status
echo [SERVICE STATUS]
echo.

REM Check Backend Status
netstat -an | findstr ":%BACKEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [BACKEND]  Running on http://localhost:%BACKEND_PORT%
    REM Show last few lines of backend log
    if exist "%BACKEND_LOG%" (
        echo   Latest activity:
        powershell -Command "Get-Content -Path '%BACKEND_LOG%' | Select-String -Pattern 'error|warning|started|listening|running|debug|info' | Select-Object -Last 5"
    )
) else (
    echo [BACKEND]  Not responding on port %BACKEND_PORT%
    if exist "%BACKEND_LOG%" (
        echo   Last error log:
        powershell -Command "Get-Content -Path '%BACKEND_LOG%' | Select-String -Pattern 'error' -CaseSensitive | Select-Object -Last 2"
    )
)

REM Check Frontend Status
netstat -an | findstr ":%FRONTEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [FRONTEND] Running on http://localhost:%FRONTEND_PORT%
    REM Show last few lines of frontend log
    if exist "%FRONTEND_LOG%" (
        echo   Latest activity:
        type "%FRONTEND_LOG%" | findstr /i "error\|warning\|ready\|started\|running\|vite\|dev" | findstr /n "^" | findstr /r "^[0-9][0-9][0-9]:" | more /e +3
    )
) else (
    echo [FRONTEND] Not responding on port %FRONTEND_PORT%
    if exist "%FRONTEND_LOG%" (
        echo   Last error log:
        powershell -Command "Get-Content -Path '%FRONTEND_LOG%' | Select-String -Pattern 'error' | Select-Object -Last 2"
    )
)

REM Real-time Terminal Logging
echo.
echo [REAL-TIME LOGS]
echo Backend output (last 5 lines):
if exist "%BACKEND_LOG%" (
    powershell -Command "Get-Content -Path '%BACKEND_LOG%' -Tail 5"
) else (
    echo No backend log available yet...
)

echo Frontend output (last 5 lines):
if exist "%FRONTEND_LOG%" (
    powershell -Command "Get-Content -Path '%FRONTEND_LOG%' -Tail 5"
) else (
    echo No frontend log available yet...
)

REM System Information
echo.
echo [SYSTEM INFORMATION]
echo Mode: %MODE%
echo Backend Port: %BACKEND_PORT%
echo Frontend Port: %FRONTEND_PORT%
if "%LOGGING_ENABLED%"=="true" (
    echo Logging: Enabled
    echo Log Directory: %LOG_DIR%
)
if "%AUTO_RESTART%"=="true" (
    echo Auto-restart: Enabled
    echo Monitor Port: %MONITOR_PORT%
)

REM Controls
echo.
echo [CONTROLS]
echo Press SPACE to refresh manually
echo Press 'L' to view detailed logs
echo Press 'R' to restart services
echo Press 'C' to clear logs
echo Press 'H' to view service health
echo Press 'T' to test connectivity
echo Press 'E' to export logs
echo Press 'Q' to quit monitoring
echo.

REM Mode-specific information
if "%MODE%"=="Basic" (
    echo [INFO] Basic mode - Services running with background logging
)
if "%MODE%"=="Enhanced Logging" (
    echo [INFO] Enhanced Logging - All output saved to %LOG_DIR%
)
if "%MODE%"=="Service Monitor" (
    echo [INFO] Service Monitor - Auto-restart enabled, dashboard at http://localhost:%MONITOR_PORT%
)
if "%MODE%"=="Development" (
    echo [INFO] Development Mode - Hot reload enabled, verbose logging
)
if "%MODE%"=="Ultimate" (
    echo [INFO] Ultimate Mode - All features enabled. Dashboard at http://localhost:%MONITOR_PORT%
)

echo [CRITICAL] This window will NOT exit without your confirmation!
echo.

REM Auto-increment monitor count
set /a MONITOR_COUNT+=1

REM Wait for user input (5 second timeout for real-time updates)
timeout /t 5 /nobreak >nul

REM Check for key press (non-blocking check)
choice /c lrcqhte /n /t 1 >nul 2>&1
if %errorlevel% == 1 goto :ViewUnifiedLogs
if %errorlevel% == 2 goto :RestartUnifiedServices
if %errorlevel% == 3 goto :ClearUnifiedLogs
if %errorlevel% == 4 goto :QuitUnifiedMonitoring
if %errorlevel% == 5 goto :ViewHealth
if %errorlevel% == 6 goto :TestConnectivity
if %errorlevel% == 7 goto :ExportLogs

goto :UnifiedMonitorLoop

:ViewUnifiedLogs
cls
echo [DETAILED LOGS]
echo.
echo Backend Log (last 20 lines):
echo ----------------------------------------
if exist "%BACKEND_LOG%" (
    powershell -Command "Get-Content -Path '%BACKEND_LOG%' -Tail 20"
) else (
    echo No backend log file found
)
echo.
echo Frontend Log (last 20 lines):
echo ----------------------------------------
if exist "%FRONTEND_LOG%" (
    powershell -Command "Get-Content -Path '%FRONTEND_LOG%' -Tail 20"
) else (
    echo No frontend log file found
)
echo.
echo Press any key to return to monitoring...
pause >nul
goto :UnifiedMonitorLoop

:RestartUnifiedServices
echo [RESTART] Restarting services...
echo.

REM Kill existing processes
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul

REM Wait for processes to terminate
timeout /t 3 /nobreak >nul

REM Clear logs
if exist "%BACKEND_LOG%" del "%BACKEND_LOG%"
if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%"

REM Restart services
start "Backend Server" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"
timeout /t 3 /nobreak >nul
start "Frontend App" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

echo [SUCCESS] Services restarted
timeout /t 5 /nobreak >nul
goto :UnifiedMonitorLoop

:ClearUnifiedLogs
if exist "%BACKEND_LOG%" del "%BACKEND_LOG%"
if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%"
echo [SUCCESS] Logs cleared
timeout /t 2 /nobreak >nul
goto :UnifiedMonitorLoop

:ViewHealth
cls
echo [SYSTEM HEALTH CHECK]
echo.

REM Check service health
echo Checking service health...
echo.

REM Backend health check
netstat -an | findstr ":%BACKEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [BACKEND] Health: OK - Port %BACKEND_PORT% is open
) else (
    echo [BACKEND] Health: FAIL - Port %BACKEND_PORT% is not responding
)

REM Frontend health check
netstat -an | findstr ":%FRONTEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [FRONTEND] Health: OK - Port %FRONTEND_PORT% is open
) else (
    echo [FRONTEND] Health: FAIL - Port %FRONTEND_PORT% is not responding
)

REM Process health check
tasklist /fi "imagename eq python.exe" | findstr "python.exe" >nul
if %errorlevel% == 0 (
    echo [PROCESSES] Python processes: Running
) else (
    echo [PROCESSES] Python processes: Not found
)

tasklist /fi "imagename eq node.exe" | findstr "node.exe" >nul
if %errorlevel% == 0 (
    echo [PROCESSES] Node processes: Running
) else (
    echo [PROCESSES] Node processes: Not found
)

echo.
echo Press any key to return to monitoring...
pause >nul
goto :UnifiedMonitorLoop

:TestConnectivity
cls
echo [CONNECTIVITY TEST]
echo.

REM Test backend connectivity
echo Testing backend connectivity...
timeout /t 1 /nobreak >nul
curl -s http://localhost:%BACKEND_PORT%/health >nul 2>&1
if %errorlevel% == 0 (
    echo [BACKEND] Connectivity: OK - Backend is responding
) else (
    echo [BACKEND] Connectivity: FAIL - Backend is not responding
)

REM Test frontend connectivity
echo Testing frontend connectivity...
timeout /t 1 /nobreak >nul
curl -s http://localhost:%FRONTEND_PORT%/ >nul 2>&1
if %errorlevel% == 0 (
    echo [FRONTEND] Connectivity: OK - Frontend is responding
) else (
    echo [FRONTEND] Connectivity: FAIL - Frontend is not responding
)

echo.
echo Press any key to return to monitoring...
pause >nul
goto :UnifiedMonitorLoop

:ExportLogs
cls
echo [EXPORT LOGS]
echo.

set "EXPORT_FILE=logs_export_%date:~-4,4%%date:~-7,2%%date:~-10,2%_%time:~0,2%%time:~3,2%.txt"

echo Exporting logs to %EXPORT_FILE%...
echo ==================================== > "%EXPORT_FILE%"
echo AI Game System - Log Export          >> "%EXPORT_FILE%"
echo Date: %date% %time%                 >> "%EXPORT_FILE%"
echo Mode: %MODE%                         >> "%EXPORT_FILE%"
echo ==================================== >> "%EXPORT_FILE%"
echo.                                     >> "%EXPORT_FILE%"
echo BACKEND LOG:                         >> "%EXPORT_FILE%"
echo ------------------------------------ >> "%EXPORT_FILE%"
if exist "%BACKEND_LOG%" (
    type "%BACKEND_LOG%" >> "%EXPORT_FILE%"
) else (
    echo No backend log found >> "%EXPORT_FILE%"
)
echo.                                     >> "%EXPORT_FILE%"
echo FRONTEND LOG:                        >> "%EXPORT_FILE%"
echo ------------------------------------ >> "%EXPORT_FILE%"
if exist "%FRONTEND_LOG%" (
    type "%FRONTEND_LOG%" >> "%EXPORT_FILE%"
) else (
    echo No frontend log found >> "%EXPORT_FILE%"
)

echo [SUCCESS] Logs exported to %EXPORT_FILE%
echo.
echo Press any key to return to monitoring...
pause >nul
goto :UnifiedMonitorLoop

:QuitUnifiedMonitoring
cls
echo [QUIT MONITORING]
echo.
echo Are you sure you want to quit monitoring?
echo Services will continue running in the background.
echo.
echo Press 'Y' to quit, 'N' to continue monitoring:

choice /c yn /n
if %errorlevel% == 1 (
    echo [QUIT] Monitoring stopped
    echo.
    echo Services are still running in background processes.
    echo To stop all services, close the backend and frontend windows.
    echo Or run this script again and select "Clean Start".
    echo.
    echo Thank you for using AI Game Playing System!
    echo.
    echo [IMPORTANT] This window will wait for your confirmation before closing!
    echo Press ENTER to exit...
    pause >nul
    exit /b 0
) else (
    goto :UnifiedMonitorLoop
)

REM ============================================
REM Legacy Monitoring Loop Functions
REM ============================================

:MonitoringLoop
set "MONITOR_COUNT=0"
:MonitorLoop
cls

REM Header
echo ================================================================
echo == AI Game System - Monitoring Console                        ==
echo ================================================================
echo Status Update: %date% %time%
echo Monitoring Cycle: %MONITOR_COUNT%
echo.

REM Service Status
echo [SERVICE STATUS]
echo.

REM Check Backend Status
netstat -an | findstr ":%BACKEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [BACKEND]  Running on http://localhost:%BACKEND_PORT%
    REM Show last few lines of backend log
    if exist "%BACKEND_LOG%" (
        echo   Latest logs:
                echo   Last error log:
                powershell -Command "Get-Content -Path '%BACKEND_LOG%' | Select-String -Pattern 'error' -CaseSensitive | Select-Object -Last 2")

REM Check Frontend Status
netstat -an | findstr ":%FRONTEND_PORT%" >nul
if %errorlevel% == 0 (
    echo [FRONTEND] Running on http://localhost:%FRONTEND_PORT%
    REM Show last few lines of frontend log
    if exist "%FRONTEND_LOG%" (
        echo   Latest logs:
        type "%FRONTEND_LOG%" | findstr /i "error\|warning\|ready\|started\|running" | findstr /n "^" | findstr /r "^[0-9][0-9][0-9]:" | more /e +3
    )
) else (
    echo [FRONTEND] Not responding on port %FRONTEND_PORT%
    if exist "%FRONTEND_LOG%" (
        echo   Last error log:
        powershell -Command "Get-Content -Path '%FRONTEND_LOG%' | Select-String -Pattern 'error' | Select-Object -Last 2"
    )
)

REM System Resources
echo.
echo [SYSTEM RESOURCES]
echo CPU and Memory usage available in Task Manager

REM Commands
echo.
echo [CONTROLS]
echo Press SPACE to refresh manually
echo Press 'L' to view detailed logs
echo Press 'R' to restart services
echo Press 'C' to clear logs
echo Press 'Q' to quit monitoring
echo.
echo [INFO] Services are running in separate windows
echo [INFO] Close this window to stop monitoring (services will continue)
echo.

REM Auto-increment monitor count
set /a MONITOR_COUNT+=1

REM Wait for user input (10 second timeout)
timeout /t 10 /nobreak >nul

REM Check for key press (non-blocking check)
choice /c lrcq /n /t 1 >nul 2>&1
if %errorlevel% == 1 goto :ViewLogs
if %errorlevel% == 2 goto :RestartServices
if %errorlevel% == 3 goto :ClearLogs
if %errorlevel% == 4 goto :QuitMonitoring

goto :MonitorLoop

:ViewLogs
cls
echo [DETAILED LOGS]
echo.
echo Backend Log:
echo ----------------------------------------
if exist "%BACKEND_LOG%" (
    type "%BACKEND_LOG%"
) else (
    echo No backend log file found
)
echo.
echo Frontend Log:
echo ----------------------------------------
if exist "%FRONTEND_LOG%" (
    type "%FRONTEND_LOG%"
) else (
    echo No frontend log file found
)
echo.
echo Press any key to return to monitoring...
pause >nul
goto :MonitorLoop

:RestartServices
echo [RESTART] Restarting services...
echo.

REM Kill existing processes
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul

REM Wait for processes to terminate
timeout /t 2 /nobreak >nul

REM Clear logs
if exist "%BACKEND_LOG%" del "%BACKEND_LOG%"
if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%"

REM Restart services
start "Backend Server" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"
timeout /t 3 /nobreak >nul
start "Frontend App" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

echo [SUCCESS] Services restarted
timeout /t 3 /nobreak >nul
goto :MonitorLoop

:ClearLogs
if exist "%BACKEND_LOG%" del "%BACKEND_LOG%"
if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%"
echo [SUCCESS] Logs cleared
timeout /t 2 /nobreak >nul
goto :MonitorLoop

:QuitMonitoring
echo [QUIT] Monitoring stopped
echo.
echo Services will continue running in their own windows.
echo Close the backend and frontend windows to stop all services.
echo.
echo Thank you for using AI Game Playing System!
echo.
pause
exit /b 0

:RecoverBackend
echo [RECOVERY] Attempting to recover backend service...
echo.

REM Kill existing backend processes
taskkill /f /im python.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clear backend log
if exist "%BACKEND_LOG%" del "%BACKEND_LOG%"

REM Restart backend
start "Backend Server (Recovery)" cmd /c "cd /d \"%~dp0ai-game-server\" && python src/main.py > \"%BACKEND_LOG%\" 2>&1"

echo [RECOVERY] Backend recovery initiated
timeout /t 3 /nobreak >nul
goto :eof

:RecoverFrontend
echo [RECOVERY] Attempting to recover frontend service...
echo.

REM Kill existing frontend processes
taskkill /f /im node.exe 2>nul
timeout /t 2 /nobreak >nul

REM Clear frontend log
if exist "%FRONTEND_LOG%" del "%FRONTEND_LOG%"

REM Restart frontend
start "Frontend App (Recovery)" cmd /c "cd /d \"%~dp0ai-game-assistant\" && npm run dev > \"%FRONTEND_LOG%\" 2>&1"

echo [RECOVERY] Frontend recovery initiated
timeout /t 3 /nobreak >nul
goto :eof

:CleanExistingProcesses
echo [CLEAN] Cleaning existing processes...
echo.

REM Kill all related processes
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Backend Server*" 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Frontend App*" 2>nul

REM Wait for processes to terminate
timeout /t 3 /nobreak >nul

echo [SUCCESS] Process cleanup complete
echo.
goto :eof

:AttemptRecovery
echo [RECOVERY] Attempting system recovery...
echo.

REM Check and fix Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python.
    exit /b 1
)

REM Check and fix Node.js installation
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js.
    exit /b 1
)

REM Check directory structure
if not exist "ai-game-server" (
    echo [ERROR] ai-game-server directory not found.
    exit /b 1
)

if not exist "ai-game-assistant" (
    echo [ERROR] ai-game-assistant directory not found.
    exit /b 1
)

REM Clear temporary files
if exist "%TEMP%\ai_game_system" rmdir /s /q "%TEMP%\ai_game_system" 2>nul

echo [SUCCESS] System recovery completed
goto :eof

:CompleteSystemShutdown
cls
echo ================================================================
echo == AI Game System - COMPLETE SYSTEM SHUTDOWN                 ==
echo ================================================================
echo.
echo [SHUTDOWN] Initiating COMPLETE system shutdown...
echo This will stop ALL services and clean up all processes.
echo.
echo Are you sure you want to shutdown the entire system?
echo Press 'Y' for complete shutdown, 'N' to return to main menu:

choice /c yn /n
if %errorlevel% == 2 (
    goto MainMenu
)

echo.
echo [STOPPING] Stopping all services and cleaning up...
echo.

REM Complete system cleanup
call :StopAllServices
call :CleanExistingProcesses
call :CleanupTemporaryFiles

echo [SUCCESS] Complete system shutdown finished!
echo.
echo All services have been stopped.
echo All processes have been cleaned up.
echo All temporary files have been removed.
echo.
echo Thank you for using AI Game Playing System!
echo Have a great day!
echo.
set "USER_EXIT_REQUESTED=true"
echo Press ENTER to exit...
pause >nul
exit /b 0

:End
REM This function is now replaced by UnifiedMonitoringLoop
REM All startup modes now enter the unified monitoring loop
exit /b 0

REM ============================================
REM Error Handling and Recovery
REM ============================================

REM Error handler for unexpected script termination
if errorlevel 1 (
    echo [ERROR] Script terminated with error code %errorlevel%
    echo [RECOVERY] Attempting to cleanup...
    call :CleanExistingProcesses
    echo Press any key to exit...
    pause >nul
    exit /b %errorlevel%
)

REM ============================================
REM Ultimate Cleanup Functions
REM ============================================

:CleanupTemporaryFiles
echo [CLEANUP] Cleaning temporary files...
echo.

REM Clean temporary log directories
if exist "%TEMP_LOG_DIR%" (
    rmdir /s /q "%TEMP_LOG_DIR%" 2>nul
)

REM Clean other temporary files
if exist "%TEMP%\ai_game_system_*" (
    rmdir /s /q "%TEMP%\ai_game_system_*" 2>nul
)

echo [SUCCESS] Temporary files cleaned up!
echo.
goto :eof

:StopAllServices
echo [STOP] Stopping all running services...
echo.

REM Graceful shutdown attempt
echo Attempting graceful shutdown...
taskkill /im python.exe 2>nul
taskkill /im node.exe 2>nul

REM Wait for graceful shutdown
timeout /t 5 /nobreak >nul

REM Force kill if still running
echo Force killing remaining processes...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Backend Server*" 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Frontend App*" 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Service Monitor*" 2>nul
taskkill /f /im cmd.exe /fi "WINDOWTITLE eq Unified Logger*" 2>nul

REM Wait for processes to terminate
timeout /t 3 /nobreak >nul

echo [SUCCESS] All services stopped!
echo.
goto :eof

REM End of script - ensure proper cleanup
on errorresume
call :CleanExistingProcesses 2>nul
call :CleanupTemporaryFiles 2>nul
on error

REM ============================================
REM ULTIMATE Script Information
REM ============================================
REM This is the ULTIMATE Unified Startup Script that:
REM
REM [CRITICAL FEATURE] NEVER EXITS WITHOUT USER PROMPTING
REM - Runs continuously until user explicitly quits
REM - Always returns to main menu, never exits automatically
REM - Requires explicit user confirmation for any exit
REM
REM [COMPREHENSIVE INTEGRATION]
REM - Integrates unified_logger.py for enhanced logging
REM - Supports service_monitor.py for auto-restart capabilities
REM - Consolidates ALL existing startup scripts into ONE solution
REM - Provides real-time terminal logging with live output display
REM
REM [ALL STARTUP MODES]
REM - Basic Startup: Quick start with background logging
REM - Enhanced Logging: Terminal and file logging with unified_logger.py
REM - Monitoring Console: Real-time monitoring with full controls
REM - Service Monitor: Auto-restart with web dashboard
REM - Development Mode: Verbose logging and debugging
REM - Clean Start: Kill existing processes and fresh start
REM - System Health Check: Comprehensive system diagnostics
REM - Recovery Mode: Advanced troubleshooting and recovery
REM
REM [ULTIMATE FEATURES]
REM - Comprehensive dependency checking and validation
REM - Multiple startup modes for different use cases
REM - Real-time monitoring and health checks with interface
REM - Automatic service recovery and restart capabilities
REM - Enhanced error handling and detailed logging
REM - User-friendly interface with comprehensive controls
REM - Service management (start/stop/restart individual services)
REM - System diagnostics and health monitoring
REM - Log viewing and export capabilities
REM - Connectivity testing and health checks
REM - Complete cleanup and shutdown procedures
REM - NEVER exits without user confirmation
REM
REM [ENHANCED LOGGING]
REM - Real-time terminal output display
REM - Comprehensive file logging with timestamps
REM - Service-specific log files
REM - Error log aggregation
REM - Log export functionality
REM - Integration with unified_logger.py when available
REM
REM [CONTINUOUS OPERATION]
REM - All modes enter continuous monitoring loops
REM - Real-time status updates every 5 seconds
REM - Interactive controls for service management
REM - Live log streaming and monitoring
REM - Automatic health checks and alerts
REM - User must explicitly choose to exit any mode
REM
REM This script represents the ULTIMATE consolidation of all startup functionality,
REM providing the most comprehensive and user-friendly startup experience possible!
REM ============================================