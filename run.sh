#!/bin/bash

# Exit if any command fails
set -e

conda activate pytorch_ddp

python fineweb.py

torchrun --standalone --nproc_per_node 8 train.py
