@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  Install numba (about 45 MB) into the Python the app uses.
REM  Model runs and calibration become about 10 times faster, for abcd and NAM.
REM  Close the app first; start it again with run_app.bat afterwards.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"

set "PY="
if exist "%~dp0.python_path.txt" set /p PY=<"%~dp0.python_path.txt"
if defined PY if not exist "%PY%" set "PY="
if not defined PY if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY set "PY=python"

echo Python used by the app:
echo     %PY%
echo.

"%PY%" -c "import numba; print('numba', numba.__version__, 'is already installed.')" 2>NUL
if not errorlevel 1 goto check

echo Downloading numba and llvmlite (about 45 MB). The download resumes after a dropped
echo connection, and running this file again continues where it stopped.
echo.
"%PY%" tools\install_numba.py
if errorlevel 1 (
    echo.
    echo The installation stopped. Run this file again to continue the download.
    pause
    exit /b 1
)

:check
echo.
echo Checking that the model's fast kernels compile (the first time takes about a minute)...
"%PY%" -c "import sys; sys.path.insert(0, '.'); from hydro import abcd, nam; print('abcd numba:', abcd.HAVE_NUMBA, ' NAM numba:', nam.HAVE_NUMBA); import numpy as np; P = np.random.rand(24, 10) * 100; E = P * 0 + 120; nam.run_nam(P, E, dict(umax=15, lmax=300, cqof=0.4, ckif=1, tof=0.3, tif=0.3, tg=0.2, ckbf=6)); abcd.run_abcd(P, E, 0.99, 700, 0.4, 0.1); print('OK: start the app with run_app.bat')"
echo.
pause
