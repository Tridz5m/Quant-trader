@echo off
REM One-time setup: virtual environment, dependencies and a config file.
cd /d "%~dp0"

REM Use the "py" launcher that python.org installs; fall back to "python".
set "PY=python"
where py >nul 2>nul && set "PY=py"
%PY% -c "import sys; assert sys.version_info >= (3, 10) and sys.maxsize > 2**32" >nul 2>nul || goto :badpython

echo Creating virtual environment in .venv ...
%PY% -m venv .venv || goto :error
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :error
if not exist config.yaml copy config.example.yaml config.yaml >nul
echo.
echo Setup complete.
echo Next: edit config.yaml, open MT5 (logged in, Algo Trading ON), then double-click run_bot.bat
echo To run commands yourself:  .venv\Scripts\python.exe -m quant_trader backtest
pause
goto :eof

:badpython
echo 64-bit Python 3.10 - 3.14 was not found. Install it from https://www.python.org/downloads/
echo and run this script again.
pause
exit /b 1

:error
echo Setup failed - see the messages above.
pause
exit /b 1
