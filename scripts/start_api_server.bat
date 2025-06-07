@echo off
setlocal

:: 设置环境变量
set PYTHONPATH=%~dp0..

:: 默认参数
set MODEL_PATH=%~dp0..\output\model
set HOST=0.0.0.0
set PORT=8000
set DEVICE=cpu

:: 解析命令行参数
:parse_args
if "%~1"=="" goto :end_parse_args
if "%~1"=="--model_path" (
    set MODEL_PATH=%~2
    shift
    shift
    goto :parse_args
)
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
if "%~1"=="--device" (
    set DEVICE=%~2
    shift
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

:: 检查模型路径
if not exist "%MODEL_PATH%\pytorch_model.bin" (
    echo 错误: 模型文件不存在于 %MODEL_PATH%
    exit /b 1
)

if not exist "%MODEL_PATH%\tokenizer.json" (
    echo 错误: 分词器文件不存在于 %MODEL_PATH%
    exit /b 1
)

if not exist "%MODEL_PATH%\config.json" (
    echo 错误: 模型配置文件不存在于 %MODEL_PATH%
    exit /b 1
)

:: 检查依赖
pip install -q fastapi uvicorn pydantic

:: 启动API服务
echo 启动中文语言模型API服务...
echo 模型路径: %MODEL_PATH%
echo 主机: %HOST%
echo 端口: %PORT%
echo 设备: %DEVICE%

python -m src.api.model_api --model_path "%MODEL_PATH%" --host %HOST% --port %PORT% --device %DEVICE%

endlocal