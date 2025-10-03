(
cd src

TRAIN_EPOCHS="10 15 20"
DROP_OTHER="0.1 0.3"
LOSS_TYPE="MPN MPN-CE"
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
for drop_other in ${DROP_OTHER} ; do
for loss_type in ${LOSS_TYPE} ; do
for m in ${M} ; do
    # execute on all datasets
    for dataset in ${dataset_prefix}*
    do
        time \
        python3 -u train.py \
            --do_train --do_eval --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
            --train_epochs ${train_epochs} --train_lr 1e-3 \
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
)
