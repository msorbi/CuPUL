(
cd src

train_epochs=5
drop_other=0.5
loss_type=MPN-CE
m=20
CURRICULUM_TRAIN_EPOCHS="1 5 10"
CURRICULUM_TRAIN_SUB_EPOCHS="1 2"
CURRICULUM_LOSS_TYPE="MPU-CE Conf-MPU Conf-MPU-CE"
CURRICULUM_TRAIN_LR="1e-5 3e-5"
tune_output="../output/hdsner/tune/curriculum"

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

mkdir -p "${tune_output}"
for curriculum_train_lr in ${CURRICULUM_TRAIN_LR} ; do
for curriculum_train_sub_epochs in ${CURRICULUM_TRAIN_SUB_EPOCHS} ; do
for curriculum_train_epochs in ${CURRICULUM_TRAIN_EPOCHS} ; do
for curriculum_loss_type in ${CURRICULUM_LOSS_TYPE} ; do
    # execute on all datasets
    for dataset in ${dataset_prefix}*
    do
        if [ `echo "${dataset}" | cut -d '-' -f 2` = 'CDBE' ] # exclude CDBE from tuning
        then
            continue
        fi
        time \
        python3 train.py \
            --pretrained_model xlm-roberta-base \
            --do_train --do_eval --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
            --train_epochs ${train_epochs} --train_lr 1e-5 \
            --drop_other ${drop_other} --drop_entity 0.0 \
            --curriculum_train_epochs ${curriculum_train_epochs} --curriculum_train_sub_epochs ${curriculum_train_sub_epochs} \
            --curriculum_train_lr ${curriculum_train_lr} \
            --self_train_epochs 0 \
            --m ${m} \
            --loss_type ${loss_type} \
            --curriculum_loss_type ${curriculum_loss_type} \
            --no_gt_output \
            --eval_on valid \
        > "${dataset}/stdout.txt" 2> "${dataset}/stderr.txt"
        find "../data" -name '*pt' -exec rm '{}' ';'
    done
    (
        cd ..
        source scripts/eval_hdsner.sh
    )
    tune_output_dir="${tune_output}/${curriculum_train_epochs}_${curriculum_train_sub_epochs}_${curriculum_loss_type}_${curriculum_train_lr}"
    mkdir -p "${tune_output_dir}"
    find ../data -name 'hdsner_report_*.json' -exec mv '{}' "${tune_output_dir}" ';'
    find ../data -name 'pred_*' -exec rm '{}' ';'
done
done
done
done
)
