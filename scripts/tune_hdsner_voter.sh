(
cd src

TRAIN_EPOCHS="5 10"
TRAIN_LR="1e-5"
DROP_OTHER="0.1 0.3 0.5"
LOSS_TYPE="MPN MPU MPN-CE MPU-CE"
M="10 20"
tune_output="../output/hdsner/tune/voter"

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
for train_epochs in ${TRAIN_EPOCHS} ; do
for train_lr in ${TRAIN_LR} ; do
for drop_other in ${DROP_OTHER} ; do
for loss_type in ${LOSS_TYPE} ; do
for m in ${M} ; do
    # execute on all datasets
    for dataset in ${dataset_prefix}*
    do
        if [ `echo "${dataset}" | cut -d '-' -f 2` = 'CDBE' ] # exclude CDBE from tuning
        then
            continue
        fi
        echo "${dataset}" | cut -d '-' -f 2 1>&2
        time \
        python3 train.py \
            --pretrained_model xlm-roberta-base \
            --do_train --do_eval --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
            --train_epochs ${train_epochs} --train_lr ${train_lr} \
            --drop_other ${drop_other} --drop_entity 0.0 \
            --curriculum_train_epochs 0 \
            --self_train_epochs 0 \
            --m ${m} \
            --loss_type ${loss_type} \
            --no_gt_output \
            --eval_on valid \
            --num_models 1 \
        > "${dataset}/stdout.txt" 2> "${dataset}/stderr.txt"
        find "../data" -name '*pt' -exec rm '{}' ';'
    done
    (
        cd ..
        source scripts/eval_hdsner.sh
    )
    tune_output_dir="${tune_output}/${train_epochs}_${drop_other}_${loss_type}_${m}"
    mkdir -p "${tune_output_dir}"
    find ../data -name 'hdsner_report_*.json' -exec mv '{}' "${tune_output_dir}" ';'
    find ../data -name 'pred_*' -exec rm '{}' ';'
done
done
done
done
done
)
