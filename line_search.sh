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
#SBATCH -p a100-4

eval "$(conda shell.bash hook)"
source ~/.bashrc
conda activate cxrpeft



gpuid=5
# export CUDA_VISIBLE_DEVICES=0
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
# torchrun --nproc_per_node=1 --master_port `expr 12380 + $gpuid` main_finetune.py \
#     --num_workers 10 \
#     --batch_size 256 \
#     --model vit_base_patch16_lora \
#     --finetune ./pretrained_weights/mae_pretrain_vit_base.pth \
#     --epochs 100 \
#     --blr 5e-3 \
#     --min_lr 1e-6 \
#     --max_lr 1 \
#     --warmup_epochs 0 \
#     --layer_decay 0.65 \
#     --weight_decay 0.05 \
#     --drop_path 0.1 \
#     --reprob 0.25 \
#     --nb_classes 5 \
#     --dist_eval \
#     --data_size 10% \
#     --data_path /scratch.global/chen8596/chexpert/chexpertchestxrays-u20210408/ \
#     --script $0 \
#     --optimizer adamw \
#     --lora_rank 8 \
#     --lora_pos attn \
#     --output_dir /scratch.global/chen8596/test \
#     --note lora \
#     --sched_name line_search \
#     --from_begin
#     #--resume /scratch.global/chen8596/random_plateauvit_base_patch16_lora_rank8_posattn_10%_adamw_200_0.00125_64_lora/model_180.pth \
#     # --eval \
#     # uncomment two lines above for inference
#python main.py --batch_size 2048 --lr 0.0001 --optimizer AdamW --scheduler LineSearch
python main.py \
    --batch_size 8192 \
    --lr 0.001 \
    --optimizer AdamW \
    --warmup_epochs 10 \
    --scheduler LineSearch \
    --epoch 1000 \
    --c1 0.0 \
    --c2 1 \
    --resume
