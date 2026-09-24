@echo off
REM Start the XAUUSD bot. Keep this window open; press Ctrl+C to stop.
cd /d "%~dp0"

REM Prefer the project's virtual environment, then the "py" launcher, then "python".
set "PY=python"
where py >nul 2>nul && set "PY=py"
if exist .venv\Scripts\python.exe set "PY=.venv\Scripts\python.exe"

:loop
%PY% -m quant_trader run
if %ERRORLEVEL%==2 (
  echo Fix the error above, then start again.
  pause
  goto :eof
)
echo Bot exited with code %ERRORLEVEL%. Restarting in 30 seconds (Ctrl+C to abort)...
timeout /t 30
goto loop
