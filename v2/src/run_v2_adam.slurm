#!/bin/bash
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --job-name=v2-adam
#SBATCH --output=v2/results/v2_adam_%j.log
#SBATCH --error=v2/results/v2_adam_%j.err
#SBATCH --account=pelphrey

module load miniforge
conda activate numpy-ann
cd /scratch/zj6nw/numpy-ann

python -u v2/src/train.py \
    --optimizer adam \
    --epochs 150 \
    --batch-size 512 \
    --patience 15 \
    --weights-out v2/results/v2_weights_adam.json
