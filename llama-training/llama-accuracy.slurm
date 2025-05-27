#!/bin/bash
#SBATCH --job-name=llama-accuracy
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # 1 task per GPU
#SBATCH --cpus-per-task=26            # CPU threads per task
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --hint=nomultithread
#SBATCH --exclusive                   # Optional: full node

module purge
module load nvhpc-hpcx-cuda12/25.5

cd $HOME/isc25_llm
source venv/bin/activate

echo "Running on $(hostname)"
nvidia-smi
lscpu | grep -E '^Socket|^Core|^CPU(s)'

# Manually control visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Fine tuning
#torchrun --nproc_per_node=4 main.py --benchmark accuracy --device-type cuda

# Running evaluation from checkpoint
torchrun --nproc_per_node=4 main.py --benchmark accuracy --device-type cuda --checkpoint $HOME/isc25_llm/checkpoints
