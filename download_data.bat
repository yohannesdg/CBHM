@echo off
setlocal
REM ---------------------------------------------------------------------------
REM  Download the public Volta data and build data\volta.npz for the model.
REM
REM  Downloads (into data\raw, resumable - just run this file again if it stops):
REM    CHIRPS v2.0 monthly rainfall 0.05 deg, Volta box    IRI Data Library   ~60 MB
REM    TerraClimate monthly Tmax and Tmin, Volta box       Univ. of Idaho     ~45 MB each
REM    HydroSHEDS v1 void-filled DEM 30 arc-sec, Africa     hydrosheds.org     48 MB
REM    Natural Earth 1:10m lakes (for Lake Volta)           naturalearthdata   2 MB
REM  Python packages: tifffile + pyshp (well under 1 MB)
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
set "LOG=%~dp0download_log.txt"

set "PY="
if exist "%~dp0.python_path.txt" set /p PY=<"%~dp0.python_path.txt"
if defined PY if not exist "%PY%" set "PY="
if not defined PY (
    echo Run setup_and_run.bat once first, so the model's Python is known.
    goto fail
)
echo Using %PY%

echo.
echo === 1/3  Small readers for the DEM and lake files ======================
"%PY%" -c "import tifffile, shapefile" 1>NUL 2>NUL
if errorlevel 1 (
    "%PY%" -m pip install --disable-pip-version-check --prefer-binary --timeout 60 --retries 10 -r requirements-prep.txt
    "%PY%" -c "import tifffile, shapefile" 1>NUL 2>NUL
    if errorlevel 1 (
        echo Could not install tifffile and pyshp. Check the internet connection and run this file again.
        goto fail
    )
) else (
    echo Already installed.
)

echo.
echo === 2/3  Downloading the data (progress and time left shown below) ======
"%PY%" prep\prep_volta.py download
if errorlevel 1 (
    echo.
    echo A download did not finish. Run download_data.bat again: finished files are kept
    echo and the DEM and lake files resume where they stopped.
    goto fail
)

echo.
echo === 3/3  Building data\volta.npz ======================================
"%PY%" prep\prep_volta.py build > "%LOG%" 2>&1
set "RC=%errorlevel%"
type "%LOG%"
if not "%RC%"=="0" (
    echo.
    echo The build stopped with an error. The messages above are saved in download_log.txt
    goto fail
)

echo.
echo Done. Restart the app (close its window, then run run_app.bat): it opens the real
echo Volta dataset automatically. The build report above is saved in download_log.txt
echo.
pause
exit /b 0

:fail
echo.
pause
exit /b 1
