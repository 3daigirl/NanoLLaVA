import torch.nn as nn
import numpy as np

def get_lr(it):
    max_lr = 6e-4
    min_lr = max_lr * 0.1   # 0.1 according to chinchilla laws
    warmup_steps = 10
    max_steps = 50
    # linearly increase the learning rate to maximum first
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    elif it > max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
    return min_lr + (max_lr - min_lr) * coeff
