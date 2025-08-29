#!/bin/bash
#SBATCH --job-name cupul
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem=8GB
#SBATCH --partition=LocalQ
#SBATCH --nodelist=kmer
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --error=cupul.err
#SBATCH --output=cupul.out

##module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate CuPUL
bash scripts/run_hdsner.sh
# time \
# python train.py --do_train --do_eval --dataset_name CoNLL2003_KB \
#         --train_epochs 1 --train_lr 2e-5 \
#         --drop_other 0.5 --drop_entity 0.0 \
#         --curriculum_train_sub_epochs 1 --curriculum_train_lr 2e-5 --curriculum_train_epochs 5 \
#         --self_train_lr 5e-7 --self_train_epochs 5 --m 20 \
        # > data/CoNLL2003_KB/output/stdout.txt 2> data/CoNLL2003_KB/output/stderr.txt
# 2>&1 | tee results/eval.txt
