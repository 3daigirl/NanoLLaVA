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
from hellaswag import render_example, iterate_examples

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
    # device = 'cpu'

print(f"Using device: {device}")

total_batch_size = 524288     # Total batch size across all gpus, 0.5 million
B = 32                  # process B batch-size per gpu
T = 1024                # sequence length
num_steps = 19073      # Total number of tokens in dataset / Total Batch Size
                        # 10B / 2**19, same as max_steps
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
# dataset = DataloaderLite('data/input.txt', B, T, ddp_rank, ngpus, )
train_dataset = DataloaderLite(B, T, ddp_rank, ngpus, 'train')
val_dataset = DataloaderLite(B, T, ddp_rank, ngpus, 'val')

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
    
raw_model = model.module if ddp else model  # to run the function model.configure_optimizer

# train model
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, betas=(0.9, 0.95), device=device)

# create log file
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "log.txt")
with open(log_file, 'w') as f:
    pass

# create tokenizer (used in inference step)
enc = tiktoken.get_encoding("gpt2")

# helper function for hellaswag eval
def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

for i in range(num_steps):
    last_step = i == num_steps - 1
    if step % 250 == 0 or last_step:
        model.eval()
        # does it make sense to reset??
        # we'll only repeat eval for first 20 * ngpus batches
        # we will never eval over the full val set
        val_dataset.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_dataset.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                val_loss_accum += loss.detach() / val_loss_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            with open(log_file, 'a') as f:
                f.write("Validation loss: {:.4f}".format(val_loss_accum.item()))
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    # code used from Andrej Karpathy's repo
    if (step % 250 == 0 or last_step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % total_num_gpus == ddp_rank
            if i % ngpus != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    # We are rewriting the generation code with a new random number generator, 
    # so as to not mess with the random seed of the model in the model's generate function
    if ((step > 0 and step % 250 == 0) or last_step):
        model.eval()
        num_return_sequences = 5
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")


    # Training
    model.train()
    optimizer.zero_grad()
    t0 = time.time()
    batch_loss = 0.0
    for step in range(num_grad_accum):
        x, y = train_dataset.next_batch()
        x, y = x.to(device), y.to(device)
        #TODO: uncomment while running on GPU
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
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
