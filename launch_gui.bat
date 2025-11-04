@echo off
REM Quick launcher for Bark Monitor GUI with monitoring

echo Starting Bark Monitor GUI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ or add it to PATH.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "gui.py" (
    echo ERROR: gui.py not found. Please run this from the Bark Monitor directory.
    pause
    exit /b 1
)

REM Create basic directories if they don't exist
if not exist logs mkdir logs
if not exist data mkdir data
if not exist recordings mkdir recordings

echo Launching GUI...
python gui.py

if errorlevel 1 (
    echo.
    echo GUI exited with an error. Check the console output above.
    echo.
    echo Common solutions:
    echo - Install dependencies: pip install -r requirements.txt
    echo - Run Windows compatibility test: python test_windows_compatibility.py
    echo - Check Python version: python --version
    pause
)