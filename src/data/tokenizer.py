import os
import json
import logging
from collections import Counter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SimpleTokenizer:
    """简单的字符级分词器，适用于中文文本"""
    
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}
        
        # 特殊标记
        self.pad_token = "[PAD]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        self.unk_token = "[UNK]"
        
        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        
        # 特殊标记的ID
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # 初始化词汇表
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
    
    def build_vocab(self, texts, min_freq=2):
        """从文本构建词汇表
        
        Args:
            texts: 文本列表
            min_freq: 最小词频
        """
        logger.info(f"从{len(texts)}个文本样本构建词汇表...")
        
        # 统计字符频率
        counter = Counter()
        for text in texts:
            counter.update(text)
        
        # 过滤低频字符并排序
        vocab_tokens = [token for token, count in counter.most_common() 
                      if count >= min_freq and token not in self.special_tokens]
        
        # 截取到指定大小
        vocab_tokens = vocab_tokens[:self.vocab_size - len(self.special_tokens)]
        
        # 构建词汇表
        for i, token in enumerate(vocab_tokens, start=len(self.special_tokens)):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        logger.info(f"词汇表构建完成，大小为{len(self.vocab)}")
    
    def tokenize(self, text):
        """将文本分词为字符列表
        
        Args:
            text: 输入文本
            
        Returns:
            tokens: 分词后的标记列表
        """
        return list(text)
    
    def encode(self, text, add_special_tokens=True):
        """将文本编码为ID列表
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记
            
        Returns:
            ids: ID列表
        """
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        """将ID列表解码为文本
        
        Args:
            ids: ID列表
            skip_special_tokens: 是否跳过特殊标记
            
        Returns:
            text: 解码后的文本
        """
        tokens = []
        for id in ids:
            if id in self.reverse_vocab:
                token = self.reverse_vocab[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_token)
        
        return "".join(tokens)
    
    def save(self, save_path):
        """保存分词器
        
        Args:
            save_path: 保存路径
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab': self.vocab,
                'vocab_size': self.vocab_size,
                'special_tokens': self.special_tokens,
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分词器已保存到{save_path}")
    
    @classmethod
    def load(cls, load_path):
        """加载分词器
        
        Args:
            load_path: 加载路径
            
        Returns:
            tokenizer: 加载的分词器
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.vocab = data['vocab']
        tokenizer.reverse_vocab = {int(id): token for token, id in data['vocab'].items()}
        
        logger.info(f"从{load_path}加载分词器，词汇表大小为{len(tokenizer.vocab)}")
        return tokenizer
    
    def __len__(self):
        return len(self.vocab)