import os
import argparse
import logging
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 导入模型和数据处理
from src.model.transformer import TransformerModel, get_model_config
from src.data.tokenizer import SimpleTokenizer
from src.data.dataset import TextDataset, create_dataloader
from src.training.trainer import Trainer

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="训练中文语言模型")
    
    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练文件路径")
    parser.add_argument("--eval_file", type=str, help="评估文件路径")
    parser.add_argument("--file_type", type=str, default="txt", choices=["txt", "jsonl"], help="文件类型")
    parser.add_argument("--cache_dir", type=str, help="缓存目录")
    parser.add_argument("--overwrite_cache", action="store_true", help="是否覆盖缓存")
    
    # 模型参数
    parser.add_argument("--model_config", type=str, default="tiny", choices=["tiny", "small", "medium", "large"], help="模型配置")
    parser.add_argument("--vocab_size", type=int, default=5000, help="词汇表大小")
    parser.add_argument("--block_size", type=int, default=128, help="序列长度")
    parser.add_argument("--stride", type=int, default=64, help="滑动窗口步长")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4, help="批量大小")
    parser.add_argument("--num_epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_steps", type=int, default=0, help="预热步数")
    parser.add_argument("--save_steps", type=int, default=1000, help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=1000, help="评估步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="日志步数")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    
    # 设备参数
    parser.add_argument("--device", type=str, default="", help="设备，留空则自动选择")
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载训练文件
    logger.info(f"加载训练文件: {args.train_file}")
    with open(args.train_file, "r", encoding="utf-8") as f:
        train_texts = [line.strip() for line in f if line.strip()]
    
    # 创建分词器
    logger.info(f"创建分词器，词汇表大小: {args.vocab_size}")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.build_vocab(train_texts)
    
    # 保存分词器
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    logger.info(f"分词器已保存到: {tokenizer_path}")
    
    # 创建数据集
    logger.info("创建训练数据集")
    train_dataset = TextDataset(
        file_path=args.train_file,
        tokenizer=tokenizer,
        block_size=args.block_size,
        stride=args.stride,
        file_type=args.file_type,
        overwrite_cache=args.overwrite_cache,
        cache_dir=args.cache_dir,
    )
    
    # 创建数据加载器
    train_dataloader = create_dataloader(train_dataset, batch_size=args.batch_size)
    
    # 创建评估数据集和加载器
    eval_dataloader = None
    if args.eval_file:
        logger.info(f"创建评估数据集: {args.eval_file}")
        eval_dataset = TextDataset(
            file_path=args.eval_file,
            tokenizer=tokenizer,
            block_size=args.block_size,
            stride=args.stride,
            file_type=args.file_type,
            overwrite_cache=args.overwrite_cache,
            cache_dir=args.cache_dir,
        )
        eval_dataloader = create_dataloader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 获取模型配置
    logger.info(f"使用模型配置: {args.model_config}")
    config = get_model_config(args.model_config)
    config.vocab_size = len(tokenizer)
    
    # 创建模型
    logger.info("创建模型")
    model = TransformerModel(config)
    
    # 创建优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # 创建学习率调度器
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.num_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir,
    )
    
    # 训练模型
    logger.info("开始训练模型")
    trainer.train()
    logger.info("训练完成")

if __name__ == "__main__":
    main()