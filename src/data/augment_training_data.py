import os
import re
import random
import logging
import argparse
import jieba
import numpy as np
from collections import Counter

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 简化的中文同义词字典
SYNONYMS = {
    "大": ["巨大", "宏大", "庞大", "伟大"],
    "小": ["微小", "细小", "渺小", "迷你"],
    "好": ["优秀", "良好", "优良", "出色"],
    "坏": ["糟糕", "恶劣", "劣质", "差劲"],
    "快": ["迅速", "敏捷", "迅捷", "飞快"],
    "慢": ["缓慢", "迟缓", "迟钝", "蹒跚"],
    "美丽": ["漂亮", "好看", "俊美", "靓丽"],
    "丑陋": ["难看", "丑恶", "丑怪", "奇丑"],
    "聪明": ["智慧", "明智", "睿智", "机智"],
    "愚蠢": ["笨拙", "愚昧", "愚钝", "愚笨"],
    "高兴": ["快乐", "欢喜", "愉悦", "欢乐"],
    "悲伤": ["忧伤", "悲痛", "悲哀", "哀伤"],
    "重要": ["关键", "主要", "核心", "关键性"],
    "次要": ["次级", "次等", "次一级", "从属"],
    "困难": ["艰难", "艰苦", "艰巨", "艰辛"],
    "容易": ["简单", "轻松", "简易", "易如反掌"],
    "开始": ["起始", "开端", "起头", "起步"],
    "结束": ["终止", "完结", "终结", "结尾"],
    "增加": ["增长", "增多", "增大", "增强"],
    "减少": ["减小", "减弱", "降低", "缩减"],
}

def load_synonyms(file_path=None):
    """加载同义词词典
    
    Args:
        file_path: 同义词词典文件路径，如果为None则使用内置词典
        
    Returns:
        synonyms: 同义词词典
    """
    if file_path is None or not os.path.exists(file_path):
        logger.info("使用内置同义词词典")
        return SYNONYMS
    
    synonyms = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                
                words = line.split(",")
                if len(words) >= 2:
                    key = words[0].strip()
                    values = [w.strip() for w in words[1:] if w.strip()]
                    if key and values:
                        synonyms[key] = values
        
        logger.info(f"从{file_path}加载了{len(synonyms)}个同义词组")
    except Exception as e:
        logger.error(f"加载同义词词典失败: {e}")
        return SYNONYMS
    
    return synonyms

def synonym_replacement(text, synonyms, n=1):
    """同义词替换
    
    Args:
        text: 输入文本
        synonyms: 同义词词典
        n: 替换次数
        
    Returns:
        augmented_text: 增强后的文本
    """
    words = list(jieba.cut(text))
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word in synonyms]))
    random.shuffle(random_word_list)
    
    num_replaced = 0
    for random_word in random_word_list:
        synonyms_list = synonyms[random_word]
        if len(synonyms_list) >= 1:
            synonym = random.choice(synonyms_list)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        
        if num_replaced >= n:
            break
    
    return "".join(new_words)

def random_insertion(text, synonyms, n=1):
    """随机插入
    
    Args:
        text: 输入文本
        synonyms: 同义词词典
        n: 插入次数
        
    Returns:
        augmented_text: 增强后的文本
    """
    words = list(jieba.cut(text))
    new_words = words.copy()
    
    for _ in range(n):
        add_word(new_words, synonyms)
    
    return "".join(new_words)

def add_word(words, synonyms):
    """添加词
    
    Args:
        words: 词列表
        synonyms: 同义词词典
    """
    synonyms_keys = list(synonyms.keys())
    random_word = random.choice(words)
    
    if random_word in synonyms:
        random_synonym = random.choice(synonyms[random_word])
    else:
        random_synonym = random.choice(synonyms_keys)
    
    random_idx = random.randint(0, len(words))
    words.insert(random_idx, random_synonym)

def random_swap(text, n=1):
    """随机交换
    
    Args:
        text: 输入文本
        n: 交换次数
        
    Returns:
        augmented_text: 增强后的文本
    """
    words = list(jieba.cut(text))
    new_words = words.copy()
    
    for _ in range(n):
        new_words = swap_word(new_words)
    
    return "".join(new_words)

def swap_word(words):
    """交换词
    
    Args:
        words: 词列表
        
    Returns:
        new_words: 交换后的词列表
    """
    if len(words) <= 1:
        return words
    
    idx1, idx2 = random.sample(range(len(words)), 2)
    words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

def random_deletion(text, p=0.1):
    """随机删除
    
    Args:
        text: 输入文本
        p: 删除概率
        
    Returns:
        augmented_text: 增强后的文本
    """
    words = list(jieba.cut(text))
    
    if len(words) == 1:
        return text
    
    new_words = []
    for word in words:
        if random.random() > p:
            new_words.append(word)
    
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words)-1)
        return words[rand_int]
    
    return "".join(new_words)

def augment_text(text, synonyms, num_aug=4):
    """增强文本
    
    Args:
        text: 输入文本
        synonyms: 同义词词典
        num_aug: 每种方法的增强次数
        
    Returns:
        augmented_texts: 增强后的文本列表
    """
    augmented_texts = []
    
    # 同义词替换
    for _ in range(num_aug):
        n_sr = max(1, int(len(list(jieba.cut(text))) * 0.1))
        augmented_texts.append(synonym_replacement(text, synonyms, n=n_sr))
    
    # 随机插入
    for _ in range(num_aug):
        n_ri = max(1, int(len(list(jieba.cut(text))) * 0.1))
        augmented_texts.append(random_insertion(text, synonyms, n=n_ri))
    
    # 随机交换
    for _ in range(num_aug):
        n_rs = max(1, int(len(list(jieba.cut(text))) * 0.1))
        augmented_texts.append(random_swap(text, n=n_rs))
    
    # 随机删除
    for _ in range(num_aug):
        augmented_texts.append(random_deletion(text, p=0.1))
    
    # 过滤重复和空文本
    augmented_texts = [t for t in augmented_texts if t and t != text]
    
    return augmented_texts

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="增强中文训练数据")
    
    # 数据参数
    parser.add_argument("--input_file", type=str, required=True, help="输入文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出文件路径")
    parser.add_argument("--synonyms_file", type=str, help="同义词词典文件路径")
    parser.add_argument("--num_aug", type=int, default=4, help="每种方法的增强次数")
    parser.add_argument("--include_original", action="store_true", help="是否包含原始文本")
    
    args = parser.parse_args()
    
    # 加载同义词词典
    synonyms = load_synonyms(args.synonyms_file)
    
    # 加载输入文件
    logger.info(f"加载输入文件: {args.input_file}")
    with open(args.input_file, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"加载了{len(texts)}个文本样本")
    
    # 增强文本
    logger.info(f"开始增强文本，每种方法增强{args.num_aug}次")
    augmented_texts = []
    
    if args.include_original:
        augmented_texts.extend(texts)
    
    for i, text in enumerate(texts):
        if i % 100 == 0:
            logger.info(f"已处理{i}/{len(texts)}个样本")
        
        aug_texts = augment_text(text, synonyms, args.num_aug)
        augmented_texts.extend(aug_texts)
    
    # 保存增强后的文本
    logger.info(f"保存增强后的文本到: {args.output_file}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        for text in augmented_texts:
            f.write(text + "\n")
    
    logger.info(f"共生成{len(augmented_texts)}个增强样本")

if __name__ == "__main__":
    main()