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

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate sls



gpuid=5
# export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"



python main.py \
    --batch_size 256 \
    --lr 1 \
    --model_name wide_resnet \
    --dataset_name cifar10 \
    --optimizer plain_SGD \
    --accum_steps 64 \
    --interval 1 \
    --scheduler LineSearch \
    --epoch 300 \
    --c1 0.9 \
    --seed 42 \
    --save_dir ./wide_resnet_line_search \