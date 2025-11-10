cd src

dataset_prefix="../data/hdsner"
rm -r ${dataset_prefix}*
for setting in `ls ../hdsner-utils/data/`
do
    source="../hdsner-utils/data/${setting}/ner_medieval_multilingual/FR/"
    if [ ! -d "${source}" ] || [ "${setting}" = "data_raw" ]
    then
        continue
    fi
    if [ "${setting}" = "supervised" ]
    then
        output_suffix="Fully"
    else
        p=`echo "${setting}" | cut -d '-' -f 2`
        output_suffix="Dict_${p}"
    fi

    # copy and format datasets
    python3 format_hdsner_datasets.py \
        "--input-dir="${source}"" \
        "--output-prefix="${dataset_prefix}"" \
        "--output-suffix="${output_suffix}""
done

# execute on all datasets
for dataset in ${dataset_prefix}*
do
    time \
    python3 train.py \
        --pretrained_model xlm-roberta-base \
        --do_train --do_eval --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
        --train_epochs 5 --train_lr 1e-5 \
        --drop_other 0.5 --drop_entity 0.0 \
        --loss_type 'MPN-CE' \
        --curriculum_train_sub_epochs 1 --curriculum_train_lr 1e-5 --curriculum_train_epochs 10 \
        --curriculum_loss_type 'Conf-MPU-CE' \
        --self_train_lr 5e-7 --self_train_epochs 0 --m 20 \
        --no_gt_output \
    > "${dataset}/stdout.txt" 2> "${dataset}/stderr.txt"
done
