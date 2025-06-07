import os
import time
import math
import logging
import torch
import torch.nn as nn
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class Trainer:
    """语言模型训练器"""
    
    def __init__(
        self,
        model,
        train_dataloader,
        eval_dataloader=None,
        optimizer=None,
        scheduler=None,
        device=None,
        num_epochs=5,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        output_dir="output",
    ):
        """
        Args:
            model: 模型
            train_dataloader: 训练数据加载器
            eval_dataloader: 评估数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            num_epochs: 训练轮数
            save_steps: 保存步数
            eval_steps: 评估步数
            logging_steps: 日志步数
            output_dir: 输出目录
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.output_dir = output_dir
        
        # 将模型移动到设备
        self.model.to(self.device)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        logger.info(f"初始化训练器，设备: {self.device}")
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        # 记录开始时间
        start_time = time.time()
        
        # 训练循环
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self._train_epoch()
            
            # 每个epoch结束后评估
            if self.eval_dataloader is not None:
                eval_loss = self.evaluate()
                logger.info(f"Epoch {epoch+1}/{self.num_epochs} 评估损失: {eval_loss:.4f}")
                
                # 保存最佳模型
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_model(os.path.join(self.output_dir, "best_model"))
                    logger.info(f"保存最佳模型，评估损失: {eval_loss:.4f}")
        
        # 保存最终模型
        self.save_model(os.path.join(self.output_dir, "final_model"))
        
        # 计算训练时间
        training_time = time.time() - start_time
        training_time_str = time.strftime("%H:%M:%S", time.gmtime(training_time))
        logger.info(f"训练完成，总用时: {training_time_str}")
    
    def _train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        
        # 进度条
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # 将批次移动到设备
            batch = batch.to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids=batch, labels=batch)
            loss = outputs["loss"]
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            # 更新进度条
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 累计损失
            epoch_loss += loss.item()
            epoch_steps += 1
            self.global_step += 1
            
            # 日志记录
            if self.global_step % self.logging_steps == 0:
                logger.info(f"步骤 {self.global_step}: 损失 = {loss.item():.4f}")
            
            # 保存模型
            if self.global_step % self.save_steps == 0:
                self.save_model(os.path.join(self.output_dir, f"checkpoint-{self.global_step}"))
            
            # 评估模型
            if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                eval_loss = self.evaluate()
                logger.info(f"步骤 {self.global_step}: 评估损失 = {eval_loss:.4f}")
                self.model.train()  # 切回训练模式
        
        # 计算平均损失
        epoch_loss /= epoch_steps
        logger.info(f"Epoch {self.epoch+1}/{self.num_epochs} 平均损失: {epoch_loss:.4f}")
    
    def evaluate(self):
        """评估模型
        
        Returns:
            eval_loss: 评估损失
        """
        logger.info("评估模型...")
        self.model.eval()
        eval_loss = 0.0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="评估"):
                batch = batch.to(self.device)
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs["loss"]
                
                eval_loss += loss.item()
                eval_steps += 1
        
        eval_loss /= eval_steps
        perplexity = math.exp(eval_loss)
        logger.info(f"评估损失: {eval_loss:.4f}, 困惑度: {perplexity:.4f}")
        
        return eval_loss
    
    def save_model(self, output_dir):
        """保存模型
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        
        # 保存模型配置
        with open(os.path.join(output_dir, "config.py"), "w", encoding="utf-8") as f:
            config = self.model.config
            f.write(f"""from src.model.transformer import TransformerConfig

config = TransformerConfig(
    vocab_size={config.vocab_size},
    hidden_size={config.hidden_size},
    num_hidden_layers={config.num_hidden_layers},
    num_attention_heads={config.num_attention_heads},
    intermediate_size={config.intermediate_size},
    hidden_dropout_prob={config.hidden_dropout_prob},
    attention_dropout_prob={config.attention_dropout_prob},
    max_position_embeddings={config.max_position_embeddings},
    type_vocab_size={config.type_vocab_size},
    initializer_range={config.initializer_range},
)
""")
        
        logger.info(f"模型已保存到{output_dir}")
    
    @classmethod
    def load_model(cls, model_class, model_path, device=None):
        """加载模型
        
        Args:
            model_class: 模型类
            model_path: 模型路径
            device: 设备
            
        Returns:
            model: 加载的模型
        """
        # 加载配置
        import sys
        import importlib.util
        
        config_path = os.path.join(model_path, "config.py")
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        config = config_module.config
        
        # 创建模型
        model = model_class(config)
        
        # 加载权重
        model_file = os.path.join(model_path, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_file, map_location="cpu"))
        
        # 移动到设备
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        logger.info(f"从{model_path}加载模型")
        return model