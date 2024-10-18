#!/bin/bash

# Exit if any command fails
set -e

# Update the system and install basic dependencies
# echo "Updating the system..."
# sudo apt update && sudo apt upgrade -y
# sudo apt install -y git wget build-essential

# Install Miniconda (Python environment management)
echo "Installing Miniconda..."
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# Initialize conda (for bash shells)
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda init bash

# Create a Python environment for PyTorch
echo "Creating a Python environment..."
conda create -y --name pytorch_ddp python=3.10
conda activate pytorch_ddp

# Install pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (Adjust CUDA version if needed)
echo "Installing PyTorch and related packages..."
pip install torch torchvision torchaudio    # no need, pip figures out automatically, --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Lightning and other training utilities
# python -m pip install pytorch-lightning==2.1.0

# deep learning packages
pip install tiktoken transformers datasets

# secondary support packages
pip install --force-reinstall requests urllib3
pip install pyyaml

# Install Distributed Training packages
pip install torchmetrics tensorboard

# Install additional utilities
pip install numpy pandas matplotlib scikit-learn tqdm

# Install NCCL for multi-GPU communication (if needed)
echo "Installing NCCL dependencies..."
sudo apt install -y libnccl2 libnccl-dev

# Install Horovod (Optional for multi-node distributed training)
# echo "Installing Horovod..."
# pip install horovod[pytorch]

# Verify installation
echo "Verifying PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__)"

echo "Verifying CUDA availability..."
python -c "print('CUDA available:', torch.cuda.is_available())"

# Set environment variables for NCCL backend
echo "Setting environment variables..."
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # Ensure the correct network interface
export OMP_NUM_THREADS=8  # Adjust based on your instance

echo "Setup complete! Activate your environment using: 'conda activate pytorch_ddp'"
