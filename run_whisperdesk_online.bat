@echo off
title WhisperDesk (Online)
cd /d "%~dp0"
if not exist .streamlit mkdir .streamlit
if exist config.toml move /y config.toml .streamlit\config.toml >nul
if not exist .venv_online\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -m venv .venv_online
  call .venv_online\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements_online.txt
) else (
  call .venv_online\Scripts\activate.bat
)
python -m streamlit run app_online.py
pause
