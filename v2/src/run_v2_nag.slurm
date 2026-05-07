#!/bin/bash
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --job-name=v2-nag
#SBATCH --output=v2/results/v2_nag_%j.log
#SBATCH --error=v2/results/v2_nag_%j.err
#SBATCH --account=pelphrey

module load miniforge
conda activate numpy-ann
cd /scratch/zj6nw/numpy-ann

python -u v2/src/train.py \
    --optimizer nag \
    --epochs 150 \
    --batch-size 512 \
    --lr 0.01 \
    --patience 15 \
    --weights-out v2/results/v2_weights_nag.json
