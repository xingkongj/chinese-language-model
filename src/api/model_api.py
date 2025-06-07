import os
import json
import logging
import argparse
from typing import Dict, List, Optional, Union

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from uvicorn import run as uvicorn_run

# 导入项目模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.transformer import TransformerModel
from src.data.tokenizer import SimpleTokenizer

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="中文语言模型API",
    description="中文语言模型的API服务，提供文本生成功能",
    version="1.0.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model = None
tokenizer = None
device = None

# 请求模型
class GenerationRequest(BaseModel):
    prompt: str = Field(..., description="生成的提示文本")
    max_length: int = Field(100, description="最大生成长度")
    temperature: float = Field(1.0, description="温度参数，控制生成的随机性")
    top_k: int = Field(50, description="Top-K采样参数")
    top_p: float = Field(0.9, description="Top-P采样参数")
    num_return_sequences: int = Field(1, description="返回的序列数量")

# 响应模型
class GenerationResponse(BaseModel):
    generated_text: Union[str, List[str]] = Field(..., description="生成的文本")
    prompt: str = Field(..., description="原始提示文本")
    generation_time: float = Field(..., description="生成时间（秒）")

@app.on_event("startup")
def startup_event():
    """应用启动时加载模型"""
    global model, tokenizer, device
    
    # 从环境变量获取模型路径
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        logger.error("未设置MODEL_PATH环境变量")
        raise RuntimeError("未设置MODEL_PATH环境变量")
    
    # 设置设备
    device_name = os.environ.get("DEVICE", "cpu")
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer_path = os.environ.get("TOKENIZER_PATH") or os.path.join(model_path, "tokenizer.json")
    logger.info(f"加载分词器: {tokenizer_path}")
    tokenizer = SimpleTokenizer.from_file(tokenizer_path)
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    model_config_path = os.path.join(model_path, "config.json")
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    
    model = TransformerModel(model_config)
    model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()
    
    logger.info("模型加载完成，API服务准备就绪")

@app.get("/")
def read_root():
    """根路径，返回API信息"""
    return {"message": "中文语言模型API服务", "status": "运行中"}

@app.get("/health")
def health_check():
    """健康检查"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "healthy"}

@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest):
    """生成文本"""
    import time
    start_time = time.time()
    
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        # 生成文本
        generated_texts = []
        for _ in range(request.num_return_sequences):
            generated = generate(
                prompt=request.prompt,
                max_length=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
            )
            generated_texts.append(generated)
        
        # 如果只返回一个序列，直接返回字符串
        if request.num_return_sequences == 1:
            generated_result = generated_texts[0]
        else:
            generated_result = generated_texts
        
        generation_time = time.time() - start_time
        
        return GenerationResponse(
            generated_text=generated_result,
            prompt=request.prompt,
            generation_time=generation_time,
        )
    except Exception as e:
        logger.error(f"生成文本时出错: {e}")
        raise HTTPException(status_code=500, detail=f"生成文本时出错: {str(e)}")

def generate(prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9):
    """生成文本
    
    Args:
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 温度参数
        top_k: top-k采样参数
        top_p: top-p采样参数
        
    Returns:
        str: 生成的文本
    """
    global model, tokenizer, device
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids]).to(device)
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            logits = outputs.logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-K采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累积超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 如果生成了结束符，停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 添加到输入序列
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # 解码生成的文本
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"全局异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"},
    )

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动中文语言模型API服务")
    
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, help="分词器路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", type=int, default=8000, help="端口号")
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu 或 cuda)")
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path
    if args.tokenizer_path:
        os.environ["TOKENIZER_PATH"] = args.tokenizer_path
    if args.device:
        os.environ["DEVICE"] = args.device
    
    # 启动服务
    uvicorn_run(
        "model_api:app",
        host=args.host,
        port=args.port,
        reload=False,
    )

if __name__ == "__main__":
    main()