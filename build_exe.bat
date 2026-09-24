@echo off
REM Build QuantTrader.exe locally (Windows, 64-bit Python 3.10-3.14).
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py"
%PY% -m pip install -r requirements.txt pyinstaller || goto :error
%PY% -m PyInstaller --noconfirm --clean packaging\QuantTrader.spec || goto :error
%PY% -c "import subprocess,sys; sys.exit(subprocess.call([r'dist\QuantTrader.exe', '--selftest']))" || goto :selftest
echo.
echo Built dist\QuantTrader.exe (self-test passed). Copy it into its own folder and double-click it.
pause
goto :eof
:selftest
echo The exe was built but its self-test failed - see dist\selftest.log
pause
exit /b 1
:error
echo Build failed - see the messages above.
pause
exit /b 1
