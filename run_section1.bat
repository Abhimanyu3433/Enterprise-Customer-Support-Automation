@echo off
REM Section 1 runner: setup venv, install, build index, run agents, evaluate. Output goes to logs\
cd /d "%~dp0"
if not exist logs mkdir logs
set LOG=logs\run.log
echo ==== START %date% %time% ==== > %LOG%

if not exist .venv\Scripts\python.exe (
  echo [1/5] Creating virtual environment >> %LOG%
  python -m venv .venv >> %LOG% 2>&1
  if not exist .venv\Scripts\python.exe "%USERPROFILE%\anaconda3\python.exe" -m venv .venv >> %LOG% 2>&1
)
set PY=.venv\Scripts\python.exe
%PY% --version >> %LOG% 2>&1

echo [2/5] Installing requirements >> %LOG%
%PY% -m pip install --upgrade pip > logs\setup.log 2>&1
%PY% -m pip install -r requirements.txt >> logs\setup.log 2>&1
echo pip exit code %errorlevel% >> %LOG%

echo [3/5] Building vector store >> %LOG%
%PY% -m src.ingestion.build_index > logs\build_index.log 2>&1
echo build_index exit code %errorlevel% >> %LOG%

findstr /r "API_KEY=." .env >nul 2>&1
if errorlevel 1 (
  echo STOPPED: add your GOOGLE_API_KEY to .env, then run this file again >> %LOG%
  goto end
)

echo [4/5] Running agent workflow demo >> %LOG%
%PY% -m src.graph.workflow > logs\workflow.log 2>&1
echo workflow exit code %errorlevel% >> %LOG%

echo [5/5] Evaluation (27 queries) >> %LOG%
%PY% -m src.evaluation.evaluate --n 27 > logs\evaluate.log 2>&1
echo evaluate exit code %errorlevel% >> %LOG%

:end
echo ==== END %date% %time% ==== >> %LOG%
type %LOG%
pause
