#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --error=eval_error.txt
#SBATCH --output=eval_output.txt
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres gpu:1
/usr/bin/true
enable_lmod
module load container_env pytorch-gpu/2.2.0
crun python evaluation.py