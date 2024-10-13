import time
import os

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import tiktoken

from gpt2_model import GPT, GPTConfig
from dataloader import TextDataloader, DataloaderLite, TextDatasetConfig
from utils import get_lr

torch.set_float32_matmul_precision('high')

# DDP multi-GPU training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ngpus = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ngpus = 1
    master_process = True
    # auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'
    #TODO: uncomment when running on GPU
    device = 'cpu'

print(f"Using device: {device}")

total_batch_size = 524288     # Total batch size across all gpus, 0.5 million
B = 16                  # process B batch-size per gpu
T = 1024                # sequence length
assert total_batch_size % (B * T * ngpus) == 0
num_grad_accum = total_batch_size // (B * T * ngpus)
if master_process:
    print("Number of gradient accumulation steps: {}".format(num_grad_accum))

# set seed
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# initialize dataset
# dataset = TextDataloader(TextDatasetConfig(
#     text_file='data/input.txt',
#     batch_size=B,
#     block_size=T
# ), ddp_rank=ddp_rank, total_gpus=ngpus)
# data_ptr = iter(dataset)
dataset = DataloaderLite(B, T, 'data/input.txt', ddp_rank, ngpus)

# create randomly initialized model
model = GPT(GPTConfig(
    # use a multiple of 2 for efficient GPU performance, can lead to 30% performance improvement
    vocab_size = 50304
))
model = model.to(device)

## Compiling the model leads to 57% faster performance
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# train model
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)

for i in range(50):
    optimizer.zero_grad()
    t0 = time.time()
    batch_loss = 0.0
    for step in range(num_grad_accum):
        x, y = dataset.next_batch()
        x, y = x.to(device), y.to(device)
        #TODO: uncomment while running on GPU
        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
        # Average is taken over gradients of all training examples in the batch. 
        # Hence divide loss by num_grad_accum to ensure mean of all training examples is taken
        loss = loss / num_grad_accum
        batch_loss += loss.detach()
        if ddp:
            model.requires_backward_grad_sync = (step == num_grad_accum - 1)
        loss.backward()
        if ddp:
            dist.all_reduce(batch_loss, op=dist.ReduceOp.AVG)
    norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # set new learning rate based on cosine decay
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    #TODO: uncomment while running on GPU
    torch.cuda.synchronize()
    t1 = time.time()
    tps = B * T * num_grad_accum * ngpus / (t1 - t0)
    print("Loss: {:6f}, norm: {}, dt: {}, tps: {}".format(batch_loss.item(), norm, t1-t0, tps))

if ddp:
    destroy_process_group()
