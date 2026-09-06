@echo off
REM Run atomik without activating the venv, from any working directory.
REM Uses `python -m` rather than .venv\Scripts\atomik.exe: the generated .exe
REM hardcodes an absolute path to python, so it breaks if this folder is moved.
setlocal
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo error: no venv at "%~dp0.venv" -- run `uv sync` in this folder first.
    exit /b 1
)
"%PY%" -m atomik_upscale %*
