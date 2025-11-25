#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --account=jusun
#SBATCH --gres=gpu:1
#SBATCH --output=exp_log/exp_%a.out
#SBATCH --error=exp_log/exp_%a.err
#SBATCH --job-name=finetune
#SBATCH -p v100,a100-4,a100-8,apollo_agate,preempt-gpu,msigpu

python main.py \
    --batch_size 256 \
    --lr 1 \
    --optimizer Prodigy \
    --scheduler None \
    --epoch 300 \
    --min_lr 1e-6 \
    --scheduler CosineWR \
    --max_lr 1 \
    --seed 42 \
    --save_dir ./test \
    --ray_tune \
    --num_samples 20