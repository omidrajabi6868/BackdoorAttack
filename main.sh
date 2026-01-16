#!/bin/bash
#SBATCH --job-name=poison_aug
#SBATCH --error=poison_aug.txt
#SBATCH --output=poison_aug.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
/usr/bin/true
enable_lmod
module load container_env pytorch-gpu/2.2.0
crun python main.py