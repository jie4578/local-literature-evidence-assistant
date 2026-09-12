@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未找到项目独立环境 .venv\Scripts\python.exe
    echo 请先按照 README.md 的安装步骤创建 .venv 并安装 requirements.txt。
    pause
    exit /b 1
)

echo 正在启动 Local Offline 文献分析系统...
".venv\Scripts\python.exe" "paper_claude.py"
set "exit_code=%ERRORLEVEL%"

if not "%exit_code%"=="0" (
    echo [错误] 程序退出，错误代码：%exit_code%
    echo 请保留本窗口中的错误信息以便排查。
    pause
)

exit /b %exit_code%
