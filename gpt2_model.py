from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import math

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_embed: 384

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.vocab_size, config.n_embed),
            h = nn.ModuleList([Block() for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x
    
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))

    def forward(self, x):
        B, T, C = x.size()
        Q, K, V = self.c_attn(x).split(self.n_embed, dim=2)
        Q = Q.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # B, nh, T, nd
        K = K.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # B, nh, T, nd
        V = V.view(B, T, self.n_head, self.n_embed // self.n_head).transpose(1, 2)  # B, nh, T, nd
        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(K.size(-1))                    # B, nh, T, T
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float=('-inf'))       # B, nh, T, T
        attn = F.softmax(attn, dim=-1)                                              # B, nh, T, T
        x = (attn @ V)                                                              # B, nh, T, nd
        x = x.transpose(1, 2).contiguous().view(B, T, C)                            # B, T, C
        x = self.c_proj(x)
        return x
