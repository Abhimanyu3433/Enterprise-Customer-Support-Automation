@echo off
REM Section 2 runner: start FastAPI + Streamlit locally, smoke-test the endpoints, check Docker. Output in logs\
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\section2.log
set PY=.venv\Scripts\python.exe
echo ==== START %date% %time% ==== > %LOG%

echo [1/4] Installing requirements >> %LOG%
%PY% -m pip install -r requirements.txt > logs\setup2.log 2>&1
echo pip exit code %errorlevel% >> %LOG%

echo [2/4] Starting API on http://localhost:8000 (Swagger: /docs) >> %LOG%
start "RAG API" cmd /k %PY% -m uvicorn api.main:app --port 8000

echo [3/4] Smoke test of /health, /query, /feedback >> %LOG%
%PY% -m api.smoke_test > logs\api_test.log 2>&1
echo smoke test exit code %errorlevel% >> %LOG%

echo [4/4] Starting chat UI on http://localhost:8501 >> %LOG%
start "RAG UI" cmd /k %PY% -m streamlit run ui/main.py --server.headless true

docker --version >> %LOG% 2>&1
echo ==== END %date% %time% ==== >> %LOG%
type %LOG%
