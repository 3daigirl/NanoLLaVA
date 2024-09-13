from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

import math

from transformers import GPT2LMHeadModel
import tiktoken

@dataclass
class GPTConfig:
    model_type: str = 'gpt2'
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_head: int = 12
    n_embed: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # share weights of lm_head and wte
        # weights are shared between both as they perform similar tasks
        print(self.lm_head.weight.size(), self.transformer.wte.weight.size())
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Sequence legnth {T} cannot exceed block size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe = self.transformer.wpe(pos)      # T, n_embed
        te = self.transformer.wte(idx)      # B, T, n_embed
        x = pe + te
        # forward pass through transformer blocks
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)    # B, T, vocab_size
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss

    def encode_tokens(self, input_string, num_repeat):
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(input_string)
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_repeat, 1)
        return tokens

    def decode_tokens(self, token_seq):
        batch_size = token_seq.size(0)
        enc = tiktoken.get_encoding(self.config.model_type)
        dec = []
        for b in range(batch_size):
            dec.append(enc.decode(token_seq[b].tolist()))
        return dec

    @torch.no_grad()
    def generate(self, input_string, max_output_tokens, num_repeat=1):
        # set seeds to replicate results
        torch.manual_seed(42)
        # torch.cuda.manual_seed(42)
        
        x = self.encode_tokens(input_string, num_repeat)

        while x.size(-1) < max_output_tokens:
            with torch.no_grad():
                logits = self(x)            # (B, T, vocab_size)
                # take logits at the last position
                logits = logits[:, -1, :]   # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=1)
                # do top-k sampling of 50
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # (B, 50)
                # select a token from the top-k probabilities
                idx = torch.multinomial(topk_probs, 1)   # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, idx)  # (B, 1)
                # append to the sequence
                x = torch.cat((x, xcol), dim=-1)

        x = self.decode_tokens(x)
        return x

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads the pretrained weights from OpenAI GPT-2 model on Huggingface"""
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        print('Loading weights from pretrained GPT: {}'.format(model_type))

        ## Create our own model with original GPT2 params
        model_config = {
            'gpt2':    dict(n_layers=12, n_head=12, n_embed=768),      # 124M params
            'gpt2-medium': dict(n_layers=24, n_head=16, n_embed=1024),  # 350M params
            'gpt2-large': dict(n_layers=36, n_head=20, n_embed=1280),   # 774M params
            'gpt2-xl': dict(n_layers=48, n_head=25, n_embed=1600)       # 1558M params
        }[model_type]
        model_config['vocab_size'] = 50257
        model_config['block_size'] = 1024
        model_config['model_type'] = model_type
        model = GPT(GPTConfig(**model_config))
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]

        # Load pretrained model
        pretrained_model = GPT2LMHeadModel.from_pretrained(model_type)
        pretrained_sd = pretrained_model.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        pretrained_keys = pretrained_sd.keys()
        pretrained_keys = [k for k in pretrained_keys if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        pretrained_keys = [k for k in pretrained_keys if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(pretrained_keys) == len(sd_keys), f"mismatched keys: {len(pretrained_keys)} != {len(sd_keys)}"
        for k in pretrained_keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                # breakpoint()
                assert pretrained_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[k].t())
            else:
                # vanilla copy over the other parameters
                assert pretrained_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(pretrained_sd[k])

        return model

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
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
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))        # B, nh, T, T
        attn = F.softmax(attn, dim=-1)                                              # B, nh, T, T
        x = (attn @ V)                                                              # B, nh, T, nd
        x = x.transpose(1, 2).contiguous().view(B, T, C)                            # B, T, C
        x = self.c_proj(x)
        return x


def main():
    # auto-detect device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    print(f"Using device: {device}")

    # create a GPT model and load pretrained weights
    model = GPT.from_pretrained('gpt2').to(device)
    
    # perform inference on the pretrained model
    generated_strings = model.generate("Hello, I'm a language model,", 50, 5)
    for string in generated_strings:
        print(string)
