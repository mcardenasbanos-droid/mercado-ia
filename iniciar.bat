@echo off
REM Investify: arranque en 1 comando (Windows). Sin llaves arranca en modo paper.
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
streamlit run app.py
