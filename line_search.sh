#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --account=jusun
#SBATCH --gres=gpu:4
#SBATCH --output=exp_log/exp_%a.out
#SBATCH --error=exp_log/exp_%a.err
#SBATCH --job-name=finetune
#SBATCH -p a100-4

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate cxrpeft



gpuid=5
# export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

python main.py \
    --batch_size 16384 \
    --lr 1 \
    --optimizer SGD \
    --warmup_epochs 0 \
    --scheduler LineSearch \
    --condition armijo \
    --epoch 200 \
    --c1 1e-2 \
    --c2 1 \
    --save_dir test

# python main.py \
#     --batch_size 16384 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --warmup_epochs 0 \
#     --scheduler LineSearch \
#     --condition wolfe \
#     --epoch 200 \
#     --c1 1e-2 \
#     --c2 1 \
#     --save_dir condition_compare


# python main.py \
#     --batch_size 8192 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --warmup_epochs 0 \
#     --scheduler LineSearch \
#     --condition armijo \
#     --epoch 200 \
#     --c1 1e-4 \
#     --c2 1 \
#     --save_dir armijo_batch_compare


# python main.py \
#     --batch_size 4096 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --warmup_epochs 0 \
#     --scheduler LineSearch \
#     --condition wolfe \
#     --epoch 200 \
#     --c1 1e-4 \
#     --c2 1 \
#     --save_dir armijo_batch_compare

# python main.py \
#     --batch_size 2048 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --warmup_epochs 0 \
#     --scheduler LineSearch \
#     --condition wolfe \
#     --epoch 200 \
#     --c1 1e-4 \
#     --c2 1 \
#     --save_dir armijo_batch_compare


# python main.py \
#     --batch_size 1024 \
#     --lr 0.001 \
#     --optimizer Adam \
#     --warmup_epochs 0 \
#     --scheduler LineSearch \
#     --condition wolfe \
#     --epoch 200 \
#     --c1 1e-4 \
#     --c2 1 \
#     --save_dir armijo_batch_compare







