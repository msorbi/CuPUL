#!/bin/bash
#SBATCH --job-name cupulrc
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=8GB
#SBATCH --partition=LocalQ
#SBATCH --nodelist=kmer
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --error=cupulrc.err
#SBATCH --output=cupulrc.out

##module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate CuPUL

time \
bash scripts/run_rcnum.sh distant-1.00
# > stdout.txt 2> stderr.txt
# 2>&1 | tee results/eval.txt
