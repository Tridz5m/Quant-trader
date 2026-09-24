@echo off
REM Start the XAUUSD bot. Keep this window open; press Ctrl+C to stop.
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
:loop
python -m quant_trader run
if %ERRORLEVEL%==2 (
  echo Fix the error above, then start again.
  pause
  goto :eof
)
echo Bot exited with code %ERRORLEVEL%. Restarting in 30 seconds (Ctrl+C to abort)...
timeout /t 30
goto loop
