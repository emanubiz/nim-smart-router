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
echo   1. minimax-m3           2. deepseek-v4-pro        3. kimi-k2.6
echo   4. nemotron-3-ultra-550b 5. qwen3.5-397b          6. nemotron-3-super-120b
echo   7. qwen3-coder-480b      8. qwen3-235b            9. deepseek-v4-flash
echo  10. glm-5.1              11. step-3.7-flash      12. step-3.5-flash
echo  13. nemotron-super-49b   14. llama-3.1-70b
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

echo  Ctrl+C or close this window to stop the server.
python server.py
if %ERRORLEVEL% NEQ 0 (
    echo  [X] pip install litellm fastapi uvicorn
    pause
)
