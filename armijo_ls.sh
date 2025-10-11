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
#SBATCH -p v100,a100-4,a100-8,apollo_agate

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
    --epoch 1000 \
    --c1 1e-2 \
    --c2 1 \
    --save_dir ./armijo_batch_compare \
    --resume














