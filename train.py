import time

import torch
import torch.nn as nn
from torch.nn import functional as F

import math
import tiktoken

from gpt2_model import GPT, GPTConfig
from dataloader import TextDataloader, TextDatasetConfig
from utils import get_lr

# auto-detect device
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
#TODO: uncomment when running on GPU
device = 'cpu'
print(f"Using device: {device}")

torch.set_float32_matmul_precision('high')

# initialize dataset
dataset = TextDataloader(TextDatasetConfig(
    text_file='data/input.txt',
    # batch_size=16,
    # block_size=1024
))
data_ptr = iter(dataset)

# create randomly initialized model
model = GPT(GPTConfig(
    # use a multiple of 2 for efficient GPU performance, can lead to 30% performance improvement
    vocab_size = 50304
))
model = model.to(device)

## Compiling the model leads to 57% faster performance
#TODO: uncomment
# model = torch.compile(model)

# train model
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)

for i in range(50):
    x, y = next(data_ptr)
    t0 = time.time()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    #TODO: uncomment while running on GPU
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # set new learning rate based on cosine decay
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    #TODO: uncomment while running on GPU
    # torch.cuda.synchronize()
    t1 = time.time()
    print("Loss: {}, norm: {}, dt: {}, tps: {}".format(loss.item(), norm, t1-t0, dataset.B*dataset.T/(t1-t0)))
