@echo off
REM One-time setup: virtual environment, dependencies and a config file.
cd /d "%~dp0"
python -m venv .venv || goto :error
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt || goto :error
if not exist config.yaml copy config.example.yaml config.yaml
echo.
echo Setup complete. Edit config.yaml, open MT5 (logged in, Algo Trading ON), then run run_bot.bat
goto :eof
:error
echo Setup failed. Make sure 64-bit Python 3.10-3.12 is installed and on PATH.
exit /b 1
