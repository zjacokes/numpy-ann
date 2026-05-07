#!/bin/bash
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=numpy-ann-v1
#SBATCH --output=v1_train_%j.log
#SBATCH --error=v1_train_%j.err
#SBATCH --account=pelphrey

module load miniforge
conda activate numpy-ann

cd /scratch/zj6nw/numpy-ann
python -u v1/src/train.py --epochs 100 --batch-size 50 --subset 0 --weights-out v1_weights_full.json
