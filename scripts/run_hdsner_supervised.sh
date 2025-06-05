#!/bin/bash
source="../hdsner-utils/data/supervised/ner_medieval_multilingual/FR/"
datadir="../data/hdsner/supervised/"

cd src
# copy and format datasets
rm -r "${datadir}"
mkdir -p "${datadir}"
python3 format_hdsner_datasets.py \
    --input-dir "${source}" \
    --output-dir "${datadir}"

# execute on all datasets
for dataset in `ls "${datadir}"`
do
    time \
    python3 train.py \
        --do_train --do_eval --dataset_name "hdsner/supervised/${dataset}/MULTICLASS" \
        --train_epochs 1 --train_lr 1e-5 \
        --drop_other 0.3 --drop_entity 0.0 \
        --curriculum_train_sub_epochs 1 --curriculum_train_lr 1e-5 --curriculum_train_epochs 5 \
        --self_train_lr 5e-7 --self_train_epochs 5 --m 20 \
    > "${datadir}/${dataset}/MULTICLASS/stdout.txt" 2> "${datadir}/${dataset}/MULTICLASS/stderr.txt"
done
