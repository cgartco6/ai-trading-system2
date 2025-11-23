@echo off
echo Setting up AI Trading System on Windows 10...

:: Check Python version
python --version
if errorlevel 1 (
    echo Python is not installed. Please install Python 3.8 or later.
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv trading_env
call trading_env\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt

:: Create necessary directories
echo Creating directories...
if not exist models mkdir models
if not exist data mkdir data
if not exist logs mkdir logs

:: Install TA-Lib (might need manual installation)
echo Note: TA-Lib might require manual installation.
echo Download from: https://github.com/TA-Lib/ta-lib-python

echo Setup completed!
echo.
echo To start the system:
echo 1. Activate virtual environment: trading_env\Scripts\activate.bat
echo 2. Start backend: python backend\app.py
echo 3. Open frontend: open frontend\index.html in browser
pause
