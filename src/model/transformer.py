import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight * x + self.bias

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # 查询、键、值的线性变换
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 输出线性变换
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # 线性变换
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 注意力得分计算
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax归一化
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 输出线性变换
        output = self.output(context_layer)
        return output

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, config):
        super(FeedForward, self).__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.activation = nn.GELU()
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.feedforward = FeedForward(config)
        self.feedforward_layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None):
        # 自注意力层
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        # 残差连接和层归一化
        hidden_states = self.attention_layernorm(hidden_states + attention_output)
        
        # 前馈网络
        feedforward_output = self.feedforward(hidden_states)
        # 残差连接和层归一化
        hidden_states = self.feedforward_layernorm(hidden_states + feedforward_output)
        
        return hidden_states

class TransformerModel(nn.Module):
    """Transformer语言模型"""
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        
        # 词嵌入
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer层
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        
        # 输出层
        self.layernorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.init_weights()
        
        logger.info(f"Model initialized with {self.num_parameters():,} parameters")
        
    def init_weights(self):
        """初始化模型权重"""
        self.token_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
    def num_parameters(self):
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())
        
    def get_input_embeddings(self):
        return self.token_embeddings
        
    def set_input_embeddings(self, embeddings):
        self.token_embeddings = embeddings
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        # 获取输入形状
        batch_size, seq_length = input_ids.size()
        
        # 位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # 注意力掩码
        if attention_mask is not None:
            # 扩展注意力掩码 [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0  # 将0变为-10000.0，1保持不变
        
        # 嵌入层
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        # 嵌入和
        embeddings = token_embeddings + position_embeddings
        hidden_states = self.dropout(embeddings)
        
        # Transformer层
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 输出层
        hidden_states = self.layernorm(hidden_states)
        
        # 语言模型预测
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # 将logits重塑为[batch_size * seq_length, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": hidden_states,
        }

class TransformerConfig:
    """Transformer配置类"""
    def __init__(
        self,
        vocab_size=5000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

def get_model_config(config_name):
    """获取预定义的模型配置"""
    if config_name == "tiny":
        return TransformerConfig(
            vocab_size=5000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            hidden_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            max_position_embeddings=512,
        )
    elif config_name == "small":
        return TransformerConfig(
            vocab_size=10000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            hidden_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            max_position_embeddings=512,
        )
    elif config_name == "medium":
        return TransformerConfig(
            vocab_size=30000,
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            hidden_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            max_position_embeddings=512,
        )
    elif config_name == "large":
        return TransformerConfig(
            vocab_size=50000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            max_position_embeddings=512,
        )
    else:
        raise ValueError(f"未知的配置名称: {config_name}")
