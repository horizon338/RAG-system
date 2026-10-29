@echo off
chcp 65001 >nul
REM ===== RAG 问答系统 一键启动 =====
echo ============================================
echo    RAG 问答系统 - 启动中...
echo ============================================
echo.

cd /d "%~dp0"

REM 启动后端
echo [1/2] 启动后端 FastAPI (端口 8000) ...
start "RAG-Backend" cmd /k "cd /d %cd% && python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

REM 启动前端
echo [2/2] 启动前端 Vue (端口 5173) ...
cd frontend
start "RAG-Frontend" cmd /k "cd /d %cd% && npm.cmd run dev"

echo.
echo ✅ 启动完成！
echo    后端 API:  http://localhost:8000  (API 文档: /docs)
echo    前端界面:  http://localhost:5173
echo.
echo 按任意键退出此窗口（后端前端继续运行）...
pause >nul
