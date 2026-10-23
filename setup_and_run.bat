@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  Volta water balance model: find a Python that can run it, then start the app.
REM
REM  1. Looks for a Python 3.10-3.13 that ALREADY has numpy, scipy, fastapi and
REM     uvicorn (python on PATH, the py launcher, Anaconda...). If one is found
REM     it is used as-is: nothing is downloaded and conda is never run.
REM  2. Otherwise it builds .venv on top of the best Python found, reusing that
REM     Python's packages and downloading only what is missing (with a log).
REM
REM  The chosen interpreter is remembered in .python_path.txt for run_app.bat.
REM
REM  Usage:  setup_and_run.bat           find/prepare Python, then run
REM          setup_and_run.bat prep      also install xarray, netCDF4, rasterio
REM                                      (needed only for prep\prep_volta.py)
REM          setup_and_run.bat rebuild   forget the choice and search again
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
set "CHOICE=%~dp0.python_path.txt"
set "VENV=%~dp0.venv"
set "LOG=%~dp0setup_log.txt"
set "PIP=-m pip install --disable-pip-version-check --prefer-binary --timeout 60 --retries 10 --log "%LOG%""
set "CORE=import numpy, scipy, fastapi, uvicorn"
echo ==== setup started %DATE% %TIME% ==== >> "%LOG%"

if /i "%~1"=="rebuild" (
    del /q "%CHOICE%" 2>NUL
    rmdir /s /q "%VENV%" 2>NUL
)

REM --- 0. a previous choice that still works -----------------------------------
set "PY="
if exist "%CHOICE%" set /p PY=<"%CHOICE%"
if defined PY (
    "%PY%" -c "%CORE%" 1>NUL 2>NUL
    if not errorlevel 1 goto chosen
    set "PY="
)

REM --- 1. a Python that already has everything ---------------------------------
echo Looking for a Python that already has the model's packages...
set "READY="
set "BASEPY="
call :scan
if defined READY (
    set "PY=%READY%"
    goto chosen
)

REM --- 2. build .venv on the best Python found, download what is missing -------
if exist "%VENV%\Scripts\python.exe" (
    set "PY=%VENV%\Scripts\python.exe"
    goto install
)
if not defined BASEPY (
    echo.
    echo No Python 3.10 to 3.13 was found. Install Python 3.11 or 3.12 from
    echo https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
    echo then run this file again.
    goto fail
)
echo No ready Python found. Creating .venv on top of:
echo     %BASEPY%
"%BASEPY%" -m venv --system-site-packages "%VENV%"
if errorlevel 1 goto fail
set "PY=%VENV%\Scripts\python.exe"

:install
echo.
echo === Downloading the missing packages (progress below; log in setup_log.txt) ===
"%PY%" %PIP% -r requirements.txt
"%PY%" -c "%CORE%" 1>NUL 2>NUL
if errorlevel 1 (
    echo.
    "%PY%" tools\check_env.py
    echo.
    echo Setup did not finish. Details are in %LOG%
    echo Run this file again to retry. It continues where it stopped.
    goto fail
)

:chosen
>"%CHOICE%" echo %PY%
echo.
"%PY%" tools\check_env.py
if /i "%~1"=="prep" (
    echo.
    echo === Installing data-preparation packages ^(xarray, netCDF4, rasterio^) ===
    "%PY%" %PIP% -r requirements-prep.txt
)
echo.
call "%~dp0run_app.bat"
exit /b %errorlevel%

REM ---------------------------------------------------------------------------
:scan
REM python on PATH first (includes the Microsoft Store Python), then py launcher, then Anaconda
for /f "delims=" %%P in ('where python 2^>NUL') do call :try "%%P"
for %%V in (3.13 3.12 3.11 3.10) do (
    for /f "usebackq delims=" %%P in (`py -%%V -c "import sys; print(sys.executable)" 2^>NUL`) do call :try "%%P"
)
for %%P in ("%USERPROFILE%\anaconda3\python.exe" "%ProgramData%\anaconda3\python.exe" "%LOCALAPPDATA%\anaconda3\python.exe" "%USERPROFILE%\miniconda3\python.exe" "%ProgramData%\miniconda3\python.exe") do call :try %%P
exit /b 0

:try
REM %1 = python.exe. READY = has the core packages; BASEPY = usable for a venv.
if defined READY exit /b 0
if not exist "%~1" exit /b 0
"%~1" -c "import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)" 1>NUL 2>NUL
if errorlevel 1 exit /b 0
if not defined BASEPY set "BASEPY=%~1"
"%~1" -c "%CORE%" 1>NUL 2>NUL
if errorlevel 1 exit /b 0
set "READY=%~1"
echo Found: %~1
exit /b 0

:fail
echo.
pause
exit /b 1
