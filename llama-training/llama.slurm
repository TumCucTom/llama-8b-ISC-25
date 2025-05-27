#!/bin/bash
#SBATCH --job-name=llama-docker
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=26
#SBATCH --time=01:00:00
#SBATCH --output=../out-logs/%x_%j.log
#SBATCH --error=../out-logs/%x_%j.log
#SBATCH --hint=nomultithread
#SBATCH --exclusive

echo "Running on $(hostname)"
nvidia-smi
lscpu | grep -E '^Socket|^Core|^CPU(s)'

# Define your project directory
PROJECT_DIR=$HOME/isc25_llm

docker run --rm --gpus all \
  --ipc=host \
  -v $(pwd):/workspace \
  -w /workspace \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "
    pip install accelerate transformers peft wandb tqdm scikit-learn numpy datasets evaluate && \
    huggingface-cli login --token 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 && \ 
    nvidia-smi && \
    lscpu | grep -E '^Socket|^Core|^CPU(s)' && \
    accelerate launch --config_file accelerate_config.yaml main.py --benchmark accuracy --precision fp8 --device-type cuda
  "
