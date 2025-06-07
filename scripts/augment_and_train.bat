@echo off
echo ===== 中文语言模型数据增强与训练脚本 =====

:: 设置环境变量
set PYTHONPATH=%~dp0..

:: 安装依赖
echo 正在安装依赖...
pip install -r ../requirements.txt

:: 创建必要的目录
if not exist "../data/augmented" mkdir "../data/augmented"
if not exist "../models" mkdir "../models"

:: 增强现有训练数据
echo 正在增强训练数据...
python ../src/data/augment_training_data.py --input_file ../data/raw/chinese_sample.txt --output_file ../data/augmented/augmented_data.txt --synonyms_file ../data/chinese_synonyms.txt --num_aug 4

:: 合并原始和增强数据
echo 正在合并原始和增强数据...
type ../data/raw/chinese_sample.txt > ../data/augmented/combined_data.txt
type ../data/augmented/augmented_data.txt >> ../data/augmented/combined_data.txt

:: 使用增强数据训练模型
echo 正在使用增强数据训练模型...
python -m src.training.train \
    --train_file ../data/augmented/combined_data.txt \
    --file_type txt \
    --model_config small \
    --vocab_size 5000 \
    --batch_size 16 \
    --num_epochs 10 \
    --learning_rate 5e-5 \
    --block_size 128 \
    --output_dir ../models/augmented_model \
    --save_steps 500 \
    --eval_steps 100 \
    --logging_steps 10 \
    --device cpu

echo 训练完成！模型已保存到 ../models/augmented_model

:: 创建文本生成脚本
echo 创建文本生成脚本...

echo @echo off > generate.bat
echo set PYTHONPATH=%%~dp0.. >> generate.bat
echo python ../generate_text.py --model_path ../models/augmented_model --prompt "%%*" --max_length 100 >> generate.bat

echo 文本生成脚本已创建，可以使用 generate.bat "你的提示文本" 来生成文本

echo ===== 全部完成 =====
pause