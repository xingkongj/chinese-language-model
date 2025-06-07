@echo off
setlocal

:: 设置环境变量
set PYTHONPATH=%~dp0..

:: 默认参数
set HOST=localhost
set PORT=8000
set PROMPT=
set MAX_LENGTH=100
set TEMPERATURE=1.0
set TOP_K=50
set TOP_P=0.9
set NUM_SEQUENCES=1

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--host" (
    set HOST=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--port" (
    set PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--prompt" (
    set PROMPT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--max_length" (
    set MAX_LENGTH=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--temperature" (
    set TEMPERATURE=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--top_k" (
    set TOP_K=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--top_p" (
    set TOP_P=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--num_return_sequences" (
    set NUM_SEQUENCES=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

:: 检查提示文本
if "%PROMPT%"=="" (
    echo 错误: 必须提供提示文本 (--prompt "你的提示文本")
    exit /b 1
)

:: 检查依赖
pip install -q requests

:: 运行API客户端
python -m src.api.api_client --host %HOST% --port %PORT% --prompt "%PROMPT%" --max_length %MAX_LENGTH% --temperature %TEMPERATURE% --top_k %TOP_K% --top_p %TOP_P% --num_return_sequences %NUM_SEQUENCES%

endlocal