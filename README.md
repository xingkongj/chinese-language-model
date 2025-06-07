# 中文语言模型

这是一个基于Transformer架构的中文语言模型项目，提供了从数据预处理、模型训练到文本生成和评估的完整流程。

## 项目结构

```
ai_language_model/
├── data/               # 数据目录
│   ├── raw/            # 原始数据
│   └── processed/      # 处理后的数据
├── notebooks/          # Jupyter笔记本
├── src/                # 源代码
│   ├── api/            # API服务
│   ├── data/           # 数据处理
│   ├── evaluation/     # 模型评估
│   ├── inference/      # 模型推理
│   ├── model/          # 模型定义
│   └── training/       # 模型训练
├── tests/              # 测试代码
├── web/                # Web界面
├── main.py             # 主入口
└── requirements.txt    # 依赖包
```

## 安装

1. 克隆仓库

```bash
git clone https://github.com/xingkongj/chinese-language-model.git
cd chinese-language-model
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 数据预处理

项目包含了一个小型的中文样本数据集，位于`data/raw/chinese_sample.txt`。您可以使用此数据集进行测试，或者添加自己的中文文本数据。

### 模型训练

训练小型中文语言模型：

```bash
# Windows
train_tiny_model.bat

# Linux/Mac
python -m src.training.train \
  --train_file data/raw/chinese_sample.txt \
  --file_type txt \
  --model_config tiny \
  --vocab_size 5000 \
  --batch_size 4 \
  --num_epochs 5 \
  --learning_rate 1e-3 \
  --block_size 128 \
  --output_dir output/tiny_chinese
```

### 文本生成

使用训练好的模型生成文本：

```bash
# Windows
generate_text.bat

# Linux/Mac
python generate_text.py \
  --model_path output/tiny_chinese \
  --prompt "自然语言处理是" \
  --max_length 100 \
  --temperature 0.7
```

### 数据增强

使用数据增强脚本生成更多训练数据：

```bash
python src/data/augment_training_data.py \
  --input_file data/raw/chinese_sample.txt \
  --output_file data/augmented/chinese_sample_augmented.txt \
  --num_aug 8
```

## 模型配置

项目提供了多种规模的模型配置：

- `tiny`: 适合快速测试和资源受限环境
- `small`: 平衡性能和资源需求
- `medium`: 提供更好的生成质量
- `large`: 最佳生成质量，但需要更多计算资源

## 贡献

欢迎提交问题和拉取请求！

## 许可证

MIT
