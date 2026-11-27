@echo off
chcp 65001 >nul
setlocal
title RAG QA System Launcher
echo ============================================
echo    RAG QA System - starting...
echo ============================================
echo.

cd /d "%~dp0"

REM ===== 1. check python =====
python --version >nul 2>&1
if errorlevel 1 goto NOPYTHON

REM ===== 2. check frontend deps =====
if exist "%~dp0frontend\node_modules" goto SKIPINSTALL
echo [INFO] frontend node_modules not found, running npm install ...
pushd "%~dp0frontend"
call npm.cmd install
popd
echo.
:SKIPINSTALL

REM ===== 3. start backend =====
echo [1/2] starting backend FastAPI on port 8000 ...
start "RAG-Backend" /d "%~dp0" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

REM ===== 4. start frontend =====
echo [2/2] starting frontend Vite on port 5173 ...
start "RAG-Frontend" /d "%~dp0frontend" cmd /k "npm.cmd run dev"

echo.
echo Done.
echo    Backend API   : http://localhost:8000
echo    API docs      : http://localhost:8000/docs
echo    Frontend UI   : http://localhost:5173
echo.
echo Open http://localhost:5173 in your browser after a few seconds.
echo Press any key to close this window (backend / frontend keep running).
pause >nul
exit /b 0

:NOPYTHON
echo [ERROR] python not found in PATH.
echo         Install Python and check "Add Python to PATH", then retry.
echo.
echo Press any key to exit...
pause >nul
exit /b 1
