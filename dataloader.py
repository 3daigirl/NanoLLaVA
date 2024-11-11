import numpy as np
import os

from dataclasses import dataclass
import tiktoken

import torch
import torch.nn as nn

from torch.utils.data import Dataset

@dataclass
class TextDatasetConfig:
    text_file: str
    batch_size: int = 4
    block_size: int = 32

class TextDataloader(Dataset):

    def __init__(self, config: TextDatasetConfig, ddp_rank, total_gpus):
        with open(config.text_file, 'r') as f:
            text = f.read()

        self.B = config.batch_size
        self.T = config.block_size
        self.ddp_rank = ddp_rank
        self.total_gpus = total_gpus

        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        print(f"Loaded {len(self.tokens)}")
        print(f"Number of batches = {len(self.tokens)/(self.B * self.T)}")

    def __len__(self):
        breakpoint()
        return len(self.tokens) - (self.B * self.T + 1)

    def __getitem__(self, idx):
        B, T = self.B, self.T
        batch = self.tokens[idx: idx + B * T + 1]
        inputs = batch[:-1].view(B, T)
        targets = batch[1:].view(B, T)
        return inputs, targets

def load_tokens_from_tokenizer(filename):
    tokens = np.load(filename)
    if isinstance(tokens, bytes):
        tokens = tokens.decode('utf-8')

    enc = tiktoken.get_encoding('gpt2')
    tokens = torch.tensor(enc.encode(tokens), dtype=torch.long)
    return tokens

def load_tokens(filename):
    tokens = np.load(filename).astype(np.int64)
    tokens = torch.tensor(tokens, dtype=torch.long)
    return tokens

class DataloaderLite:
    def __init__(self, B, T, ddp_rank, total_gpus, split='train'):
        
        ## Use this to load shakespeare data from .txt file
        # with open(text_file, 'r') as f:
            # text = f.read()

        self.B = B
        self.T = T
        self.ddp_rank = ddp_rank
        self.total_gpus = total_gpus

        # Load numpy data for FineWeb-Edu
        # txt = np.load(filename)

        root_dir = "edu_fineweb10B"
        shards = [os.path.join(root_dir, s) for s in os.listdir(root_dir) if split in s]
        self.shards = sorted(shards)
        assert len(self.shards) > 0, f"{split} doesn't have any shards"

        self.reset()

    def reset(self):
        # track the shard pointer
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        # track data pointer in this shard
        self.current_position = self.ddp_rank * self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        batch = self.tokens[self.current_position: self.current_position + B * T + 1]
        inputs = batch[:-1].view(B, T)
        targets = batch[1:].view(B, T)
        self.current_position += B * T * self.total_gpus
        # if current position exceeds this shard, proceed to next shard
        if self.current_position + (B * T * self.total_gpus + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.ddp_rank
        return inputs, targets
