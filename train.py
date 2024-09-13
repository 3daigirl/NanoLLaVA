import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import tiktoken

from gpt2_model import GPT, GPTConfig
from dataloader import TextDataloader, TextDatasetConfig

# auto-detect device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
device = 'cpu'
print(f"Using device: {device}")

# initialize dataset
dataset = TextDataloader(TextDatasetConfig(text_file='data/input.txt'))

# create randomly initialized model
model = GPT(GPTConfig())
model.to(device)

# train model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for x, y in dataset:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    _, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print("Loss: {}".format(loss.item()))
