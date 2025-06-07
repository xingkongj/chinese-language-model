import os
import torch
import logging
import numpy as np
from torch.utils.data import Dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """文本数据集，用于语言模型训练"""
    
    def __init__(
        self,
        file_path,
        tokenizer,
        block_size=128,
        stride=64,
        file_type="txt",
        overwrite_cache=False,
        cache_dir=None,
    ):
        """
        Args:
            file_path: 文本文件路径
            tokenizer: 分词器
            block_size: 最大序列长度
            stride: 滑动窗口步长
            file_type: 文件类型，支持txt和jsonl
            overwrite_cache: 是否覆盖缓存
            cache_dir: 缓存目录
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.stride = stride
        self.file_type = file_type
        
        # 缓存设置
        cache_dir = cache_dir if cache_dir else os.path.dirname(file_path)
        os.makedirs(cache_dir, exist_ok=True)
        cached_file = os.path.join(
            cache_dir,
            f"cached_lm_{os.path.basename(file_path)}_{tokenizer.vocab_size}_{block_size}_{stride}",
        )
        
        # 如果缓存存在且不覆盖，则加载缓存
        if os.path.exists(cached_file) and not overwrite_cache:
            logger.info(f"从缓存加载数据集: {cached_file}")
            self.examples = torch.load(cached_file)
            return
        
        logger.info(f"创建数据集: {file_path}")
        
        # 加载文本
        self.texts = self.load_texts(file_path, file_type)
        logger.info(f"加载了{len(self.texts)}个文本样本")
        
        # 处理文本
        self.examples = self.process_texts()
        
        # 保存缓存
        logger.info(f"将数据集保存到缓存: {cached_file}")
        torch.save(self.examples, cached_file)
    
    def load_texts(self, file_path, file_type):
        """加载文本
        
        Args:
            file_path: 文件路径
            file_type: 文件类型
            
        Returns:
            texts: 文本列表
        """
        texts = []
        
        if file_type == "txt":
            with open(file_path, "r", encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
        elif file_type == "jsonl":
            import json
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if "text" in data:
                            texts.append(data["text"].strip())
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
        
        return texts
    
    def process_texts(self):
        """处理文本
        
        Returns:
            examples: 处理后的样本
        """
        examples = []
        
        for text in self.texts:
            # 编码文本
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # 如果文本太短，直接跳过
            if len(token_ids) <= 1:
                continue
            
            # 使用滑动窗口创建样本
            for i in range(0, max(1, len(token_ids) - self.block_size + 1), self.stride):
                input_ids = token_ids[i : i + self.block_size]
                
                # 如果样本太短，进行填充
                if len(input_ids) < self.block_size:
                    input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
                
                examples.append(torch.tensor(input_ids, dtype=torch.long))
        
        logger.info(f"创建了{len(examples)}个训练样本")
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def create_dataloader(dataset, batch_size, shuffle=True):
    """创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批量大小
        shuffle: 是否打乱数据
        
    Returns:
        dataloader: 数据加载器
    """
    from torch.utils.data import DataLoader
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # 多进程加载可能在某些环境中导致问题
    )