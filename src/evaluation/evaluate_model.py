import os
import re
import json
import math
import logging
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import DataLoader

# 导入项目模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.transformer import TransformerModel, get_model_config
from src.data.tokenizer import SimpleTokenizer
from src.data.dataset import TextDataset, create_dataloader

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def calculate_perplexity(model, dataloader, device):
    """计算困惑度
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        
    Returns:
        float: 困惑度
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算困惑度"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids)
            logits = outputs.logits
            
            # 计算交叉熵损失
            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item()
            total_tokens += shift_labels.ne(0).sum().item()  # 不计算padding token
    
    # 计算困惑度: exp(平均损失)
    perplexity = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
    return perplexity

def calculate_bleu(references, hypotheses, max_n=4):
    """计算BLEU分数
    
    Args:
        references: 参考文本列表
        hypotheses: 生成文本列表
        max_n: 最大n-gram
        
    Returns:
        float: BLEU分数
    """
    # 计算n-gram精确率
    def calculate_precision(reference, hypothesis, n):
        ref_ngrams = Counter([tuple(reference[i:i+n]) for i in range(len(reference)-n+1)])
        hyp_ngrams = Counter([tuple(hypothesis[i:i+n]) for i in range(len(hypothesis)-n+1)])
        
        # 计算匹配的n-gram数量
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values()) or 1  # 避免除零
        
        return matches / total
    
    # 计算简化版BLEU分数
    def calculate_simple_bleu(reference, hypothesis, max_n):
        # 如果假设为空，返回0
        if len(hypothesis) == 0:
            return 0
        
        # 计算各n-gram精确率
        precisions = []
        for n in range(1, max_n + 1):
            if len(hypothesis) >= n and len(reference) >= n:
                precisions.append(calculate_precision(reference, hypothesis, n))
            else:
                precisions.append(0)
        
        # 几何平均
        if all(p == 0 for p in precisions):
            return 0
        
        # 过滤掉零值
        non_zero_precisions = [p for p in precisions if p > 0]
        if not non_zero_precisions:
            return 0
        
        # 计算几何平均
        log_avg = sum(math.log(p) for p in non_zero_precisions) / len(non_zero_precisions)
        
        # 简化版长度惩罚
        brevity_penalty = 1.0
        if len(hypothesis) < len(reference):
            brevity_penalty = math.exp(1 - len(reference) / len(hypothesis))
        
        return brevity_penalty * math.exp(log_avg)
    
    # 对所有参考和假设计算BLEU
    bleu_scores = []
    for ref, hyp in zip(references, hypotheses):
        bleu_scores.append(calculate_simple_bleu(ref, hyp, max_n))
    
    # 返回平均BLEU分数
    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, top_p=0.9, device="cpu"):
    """生成文本
    
    Args:
        model: 模型
        tokenizer: 分词器
        prompt: 提示文本
        max_length: 最大生成长度
        temperature: 温度参数
        top_k: top-k采样参数
        top_p: top-p采样参数
        device: 设备
        
    Returns:
        str: 生成的文本
    """
    model.eval()
    
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

def evaluate_generation(model, tokenizer, test_data, num_samples=10, max_length=100, device="cpu"):
    """评估生成质量
    
    Args:
        model: 模型
        tokenizer: 分词器
        test_data: 测试数据
        num_samples: 样本数量
        max_length: 最大生成长度
        device: 设备
        
    Returns:
        dict: 评估结果
    """
    # 随机选择样本
    if len(test_data) > num_samples:
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        samples = [test_data[i] for i in indices]
    else:
        samples = test_data
    
    # 生成文本并计算BLEU分数
    references = []
    hypotheses = []
    generated_pairs = []
    
    for text in tqdm(samples, desc="生成文本"):
        # 取前20个字符作为提示
        prompt_length = min(20, len(text) // 3)
        prompt = text[:prompt_length]
        
        # 生成文本
        generated = generate_text(
            model, tokenizer, prompt, max_length=max_length, device=device
        )
        
        # 记录结果
        reference = tokenizer.tokenize(text[prompt_length:])
        hypothesis = tokenizer.tokenize(generated[len(prompt):])
        
        references.append(reference)
        hypotheses.append(hypothesis)
        generated_pairs.append({
            "prompt": prompt,
            "reference": text[prompt_length:],
            "generated": generated[len(prompt):]
        })
    
    # 计算BLEU分数
    bleu_1 = calculate_bleu(references, hypotheses, max_n=1)
    bleu_2 = calculate_bleu(references, hypotheses, max_n=2)
    bleu_4 = calculate_bleu(references, hypotheses, max_n=4)
    
    return {
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_4": bleu_4,
        "generated_samples": generated_pairs
    }

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="评估中文语言模型")
    
    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, help="分词器路径，如果为空则使用模型路径下的tokenizer.json")
    
    # 数据参数
    parser.add_argument("--test_file", type=str, required=True, help="测试文件路径")
    parser.add_argument("--file_type", type=str, default="txt", choices=["txt", "json"], help="文件类型")
    
    # 评估参数
    parser.add_argument("--batch_size", type=int, default=16, help="批量大小")
    parser.add_argument("--block_size", type=int, default=128, help="序列长度")
    parser.add_argument("--num_samples", type=int, default=10, help="生成评估的样本数量")
    parser.add_argument("--max_length", type=int, default=100, help="最大生成长度")
    parser.add_argument("--output_file", type=str, help="评估结果输出文件")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="cpu", help="设备 (cpu 或 cuda)")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 加载分词器
    tokenizer_path = args.tokenizer_path or os.path.join(args.model_path, "tokenizer.json")
    logger.info(f"加载分词器: {tokenizer_path}")
    tokenizer = SimpleTokenizer.from_file(tokenizer_path)
    
    # 加载模型
    logger.info(f"加载模型: {args.model_path}")
    model_config_path = os.path.join(args.model_path, "config.json")
    with open(model_config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)
    
    model = TransformerModel(model_config)
    model.load_state_dict(torch.load(os.path.join(args.model_path, "pytorch_model.bin"), map_location=device))
    model.to(device)
    model.eval()
    
    # 加载测试数据
    logger.info(f"加载测试数据: {args.test_file}")
    test_dataset = TextDataset(
        file_path=args.test_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        file_type=args.file_type,
    )
    
    test_dataloader = create_dataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 计算困惑度
    logger.info("计算困惑度...")
    perplexity = calculate_perplexity(model, test_dataloader, device)
    logger.info(f"困惑度: {perplexity:.4f}")
    
    # 评估生成质量
    logger.info("评估生成质量...")
    generation_results = evaluate_generation(
        model, tokenizer, test_dataset.examples, 
        num_samples=args.num_samples, max_length=args.max_length, device=device
    )
    
    logger.info(f"BLEU-1: {generation_results['bleu_1']:.4f}")
    logger.info(f"BLEU-2: {generation_results['bleu_2']:.4f}")
    logger.info(f"BLEU-4: {generation_results['bleu_4']:.4f}")
    
    # 输出评估结果
    results = {
        "perplexity": perplexity,
        "bleu_1": generation_results["bleu_1"],
        "bleu_2": generation_results["bleu_2"],
        "bleu_4": generation_results["bleu_4"],
        "generated_samples": generation_results["generated_samples"],
    }
    
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"评估结果已保存到: {args.output_file}")
    
    # 打印生成样本
    logger.info("\n生成样本:")
    for i, sample in enumerate(results["generated_samples"][:5]):
        logger.info(f"\n样本 {i+1}:")
        logger.info(f"提示: {sample['prompt']}")
        logger.info(f"参考: {sample['reference'][:100]}...")
        logger.info(f"生成: {sample['generated'][:100]}...")

if __name__ == "__main__":
    main()