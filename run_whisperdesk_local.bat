@echo off
title WhisperDesk (Offline)
cd /d "%~dp0"
if not exist .streamlit mkdir .streamlit
if exist config.toml move /y config.toml .streamlit\config.toml >nul
if not exist .venv_local\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -m venv .venv_local
  call .venv_local\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements_local.txt
) else (
  call .venv_local\Scripts\activate.bat
)
python -m streamlit run app_local.py
pause
