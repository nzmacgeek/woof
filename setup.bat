@echo off
REM Bark Monitor Setup Script for Windows 11

echo ============================================================
echo   BARK MONITOR - Enhanced Dog Barking Detection System
echo                    Windows 11 Setup Script
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo Found Python %PYTHON_VERSION%

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
if "%SCRIPT_DIR:~-1%"=="\" set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo Setting up in directory: %SCRIPT_DIR%
cd /d "%SCRIPT_DIR%"

echo.
echo Checking for pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please reinstall Python with pip support
    pause
    exit /b 1
)

echo.
echo Creating virtual environment...
if not exist venv (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing Python dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo Creating necessary directories...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist recordings mkdir recordings
if not exist reports mkdir reports

echo.
echo Copying default configuration...
if not exist config.json (
    copy config\default.json config.json
    echo Default configuration copied to config.json
) else (
    echo Configuration file already exists
)

echo.
echo Testing installation...
python -c "import pyaudio, librosa, numpy, sklearn; print('Core dependencies available')"
if errorlevel 1 (
    echo WARNING: Some dependencies may not be properly installed
)

echo.
echo ============================================================
echo                   INSTALLATION COMPLETE
echo ============================================================
echo.
echo Quick Start Guide:
echo.
echo 1. Test the system:
echo    python setup.py
echo.
echo 2. Start monitoring:
echo    woof.bat monitor
echo.
echo 3. Launch GUI:
echo    python gui.py
echo.
echo 4. View help:
echo    woof.bat help
echo.
echo Windows-specific notes:
echo - FFmpeg is recommended for MP3 encoding
echo - Grant microphone permissions in Windows Privacy settings
echo - Run as Administrator if you encounter permission issues
echo.
echo For detailed instructions, see README.md
echo.
pause