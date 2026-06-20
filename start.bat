@echo off
title nim-smart-router — NIM Auto-Fallback Proxy
cd /d "%~dp0"
chcp 65001 >nul 2>&1

echo ================================================================
echo   nim-smart-router  --  NIM Auto-Fallback Proxy
echo   Coding-agent ready  ^|  model: "auto"  ^|  localhost:4000
echo ================================================================
echo.
echo  Models (by priority):
echo   1. kimi-k2.6       2. deepseek-v4-pro     3. deepseek-v4-flash
echo   4. minimax-m3      5. glm-5.1             6. step-3.5-flash
echo   7. step-3.7-flash  8. nemotron-super-49b  9. llama-3.1-70b
echo.

REM Load .env (skip comment lines starting with #)
if exist ".env" (
    for /f "usebackq eol=# tokens=*" %%a in (".env") do set "%%a"
)

if "%NVIDIA_NIM_API_KEY%"=="" (
    echo  [!] NVIDIA_NIM_API_KEY missing
    echo     Create .env: NVIDIA_NIM_API_KEY=nvapi-...  (https://build.nvidia.com)
    echo.
)

echo  Starting on http://127.0.0.1:4000/v1/chat/completions
echo.
python server.py
if %ERRORLEVEL% NEQ 0 (
    echo  [X] pip install litellm fastapi uvicorn
    pause
)
