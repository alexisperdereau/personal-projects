import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :].to(x.device)

# Attention Mechanism
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.qkv_linear = nn.Linear(d_model, 3 * d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv_linear(x).reshape(batch_size, seq_len, self.num_heads, 3 * self.d_k)
        q, k, v = qkv.split(self.d_k, dim=-1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.reshape(batch_size, seq_len, d_model)
        return self.output_linear(context)

# Feedforward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.feedforward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)
        ff_output = self.feedforward(x)
        return self.norm2(x + ff_output)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, dim_feedforward, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dim_feedforward) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
