import os
import argparse
import logging
import torch
import torch.nn.functional as F

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 导入模型和分词器
from src.model.transformer import TransformerModel
from src.data.tokenizer import SimpleTokenizer
from src.training.trainer import Trainer

def generate_text(
    model,
    tokenizer,
    prompt,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.0,
    device="cpu",
):
    """生成文本
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 温度参数，控制生成的随机性
        top_k: 只考虑概率最高的k个词
        top_p: 只考虑概率累积达到p的词
        repetition_penalty: 重复惩罚参数
        device: 设备
        
    Returns:
        generated_text: 生成的文本
    """
    model.eval()
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # 生成参数
    generated = input_ids
    past = None
    
    # 记录已生成的标记，用于重复惩罚
    generated_tokens = []
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(input_ids=generated[:, -1:] if past else generated)
            logits = outputs["logits"]
            past = outputs.get("past", None)
            
            # 获取最后一个时间步的logits
            next_token_logits = logits[:, -1, :]
            
            # 重复惩罚
            if repetition_penalty > 1.0:
                for token in generated_tokens:
                    next_token_logits[:, token] /= repetition_penalty
            
            # 温度缩放
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Top-K采样
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除概率累积超过阈值的标记
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过阈值的标记
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = float("-inf")
            
            # 采样下一个标记
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # 如果生成了EOS标记，则停止生成
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # 添加到已生成的序列
            generated = torch.cat((generated, next_token), dim=1)
            generated_tokens.append(next_token.item())
    
    # 解码生成的文本
    generated_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    
    return generated_text

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="使用中文语言模型生成文本")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--prompt", type=str, default="", help="提示文本")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K采样参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="重复惩罚参数")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="", help="设备，留空则自动选择")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer_path = os.path.join(args.model_path, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        # 尝试在父目录中查找
        tokenizer_path = os.path.join(os.path.dirname(args.model_path), "tokenizer.json")
    
    logger.info(f"加载分词器: {tokenizer_path}")
    tokenizer = SimpleTokenizer.load(tokenizer_path)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model = Trainer.load_model(TransformerModel, args.model_path, device=device)
    
    # 生成文本
    logger.info("开始生成文本")
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device,
    )
    
    # 打印生成的文本
    print("\n生成的文本:")
    print("-" * 50)
    print(f"{args.prompt}{generated_text}")
    print("-" * 50)

if __name__ == "__main__":
    main()