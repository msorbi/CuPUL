#!/bin/bash
if [ $# -ne 1 ] || ([ "$1" != "supervised" ] && [ "$1" != "distant" ])
then
    echo "usage: $0 (supervised|distant)"
    exit 1
fi

setting="$1"
source="../hdsner-utils/data/${setting}/ner_medieval_multilingual/FR/"
dataset_prefix="../data/hdsner-${setting}"

cd src

# copy and format datasets
rm -r ${dataset_prefix}*
# mkdir -p "${datadir}"
python3 format_hdsner_datasets.py \
    --input-dir "${source}" \
    --output-prefix "${dataset_prefix}"

# execute on all datasets
for dataset in ${dataset_prefix}*
do
    time \
    python3 train.py \
        --do_train --do_eval --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
        --train_epochs 1 --train_lr 1e-5 \
        --drop_other 0.3 --drop_entity 0.0 \
        --curriculum_train_sub_epochs 1 --curriculum_train_lr 1e-5 --curriculum_train_epochs 5 \
        --self_train_lr 5e-7 --self_train_epochs 5 --m 20 \
        --no_gt_output \
    > "${dataset}/stdout.txt" 2> "${dataset}/stderr.txt"
done
