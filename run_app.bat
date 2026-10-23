@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  Start the Volta model server and open it in the browser.
REM  Uses the Python chosen by setup_and_run.bat (.python_path.txt), then .venv,
REM  then a conda env called volta-hydro, then python on PATH.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
set "PORT=8000"

set "PY="
if exist "%~dp0.python_path.txt" set /p PY=<"%~dp0.python_path.txt"
if defined PY if not exist "%PY%" set "PY="
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\anaconda3\envs\volta-hydro\python.exe" set "PY=%USERPROFILE%\anaconda3\envs\volta-hydro\python.exe"
if not defined PY set "PY=python"

"%PY%" -c "import numpy, scipy, fastapi, uvicorn" 1>NUL 2>NUL
if errorlevel 1 (
    echo The model's Python packages are not available for:
    echo     %PY%
    echo Run setup_and_run.bat first.
    echo.
    pause
    exit /b 1
)

REM Is something already using the port? (e.g. the app is already open)
netstat -ano | findstr /r /c:":%PORT% .*LISTENING" 1>NUL 2>NUL
if not errorlevel 1 (
    echo Port %PORT% is already in use. The app may already be running:
    echo     http://localhost:%PORT%
    start "" "http://localhost:%PORT%"
    pause
    exit /b 0
)

REM Open the browser once the server answers (it loads the dataset first)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command ^
  "for($i=0;$i -lt 90;$i++){try{Invoke-WebRequest -UseBasicParsing http://127.0.0.1:%PORT%/api/status -TimeoutSec 2 | Out-Null; Start-Process 'http://localhost:%PORT%'; break}catch{Start-Sleep -Milliseconds 700}}"

echo Volta model running at http://localhost:%PORT%   (close this window to stop it)
echo Using %PY%
"%PY%" -m uvicorn backend.app:app --host 127.0.0.1 --port %PORT%
if errorlevel 1 (
    echo.
    echo The server stopped with an error. See the messages above.
    pause
)
