@echo off
REM Dog Bark Monitor Launcher Script for Windows

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0

REM Remove trailing backslash if present
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Activate virtual environment if it exists
if exist "%SCRIPT_DIR%\venv\Scripts\activate.bat" (
    call "%SCRIPT_DIR%\venv\Scripts\activate.bat"
)

REM Change to project directory
cd /d "%SCRIPT_DIR%"

REM Check if running with arguments
if "%~1"=="" (
    echo Dog Bark Detection and Monitoring System
    echo ========================================
    echo.
    echo Usage: %0 [command] [options]
    echo.
    echo Available commands:
    echo   monitor         Start bark monitoring
    echo   monitor-sim     Start in simulation mode
    echo   test            Run system self-test
    echo   stats           Show database statistics
    echo   report          Generate evidence report
    echo   list-devices    List audio devices
    echo   help            Show detailed help
    echo.
    echo Examples:
    echo   %0 monitor
    echo   %0 monitor-sim
    echo   %0 stats
    echo   %0 report --days 30
    echo.
    exit /b 1
)

REM Handle special commands
if /i "%1"=="monitor-sim" (
    shift
    python src\cli.py monitor --simulate %*
    goto :eof
)

if /i "%1"=="help" (
    python src\cli.py --help
    goto :eof
)

REM Run with all arguments
python src\cli.py %*