@echo off
title WhisperDesk (Both modes)
cd /d "%~dp0"
if not exist .streamlit mkdir .streamlit
if exist config.toml move /y config.toml .streamlit\config.toml >nul
if not exist .venv_all\Scripts\python.exe (
  echo Creating Python virtual environment...
  py -m venv .venv_all
  call .venv_all\Scripts\activate.bat
  python -m pip install --upgrade pip
  pip install -r requirements_all.txt
) else (
  call .venv_all\Scripts\activate.bat
)
python -m streamlit run app.py
pause
