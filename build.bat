@echo off
chcp 65001 >nul
echo ============================================================
echo llm-bench 打包工具
echo ============================================================
echo.

cd /d "%~dp0\.."
python build\build.py

echo.
pause
