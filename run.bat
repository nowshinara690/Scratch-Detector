@echo off
setlocal
cd /d "%~dp0"

set "REBUILD_VENV="
if exist ".venv" (
  if not exist ".venv\Scripts\python.exe" set "REBUILD_VENV=1"
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" --version >nul 2>&1
    if errorlevel 1 set "REBUILD_VENV=1"
  )
)
if defined REBUILD_VENV (
  echo Recreating the invalid virtual environment...
  rmdir /s /q ".venv"
  if exist ".venv" goto :venv_error
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3 -m venv ".venv"
  if errorlevel 1 goto :venv_error
)

".venv\Scripts\python.exe" -m streamlit --version >nul 2>&1
if errorlevel 1 (
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run app.py
exit /b %errorlevel%

:venv_error
echo Could not create the virtual environment. Install Python 3 and ensure the py launcher is available.
exit /b 1
