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


class DataloaderLite:
    def __init__(self, text_file, B, T, ddp_rank, total_gpus):
        with open(text_file, 'r') as f:
            text = f.read()

        self.B = B
        self.T = T
        self.ddp_rank = ddp_rank
        self.total_gpus = total_gpus

        enc = tiktoken.get_encoding('gpt2')
        self.tokens = torch.tensor(enc.encode(text))
        print(f"Loaded {len(self.tokens)}")
        print(f"Number of batches = {len(self.tokens)/(self.B * self.T)}")

        # track data pointer in this dataloader
        self.current_position = self.ddp_rank * self.B * self.T


    def next_batch(self):
        B, T = self.B, self.T
        batch = self.tokens[self.current_position: self.current_position + B * T + 1]
        inputs = batch[:-1].view(B, T)
        targets = batch[1:].view(B, T)
        self.current_position += B * T * self.total_gpus
        if self.current_position + (B * T * self.total_gpus + 1) > len(self.tokens):
            self.current_position = B * T * self.total_gpus
        return inputs, targets
