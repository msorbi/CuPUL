(
cd src

train_epochs=15
drop_other=0.1
loss_type=MPN
m=20
CURRICULUM_TRAIN_EPOCHS="15 20 25"
CURRICULUM_LOSS_TYPE="MPN MPN-CE MPU MPU-CE Conf-MPU Conf-MPU-CE"
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
for curriculum_train_epochs in ${CURRICULUM_TRAIN_EPOCHS} ; do
for curriculum_loss_type in ${CURRICULUM_LOSS_TYPE} ; do
    # execute on all datasets
    for dataset in ${dataset_prefix}*
    do
        time \
        python3 train.py \
            --do_train --do_eval --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
            --train_epochs ${train_epochs} --train_lr 1e-3 \
            --drop_other ${drop_other} --drop_entity 0.0 \
            --curriculum_train_epochs ${curriculum_train_epochs} --curriculum_train_sub_epochs 1 \
            --curriculum_train_lr 1e-3 \
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
    tune_output_dir="${tune_output}/${curriculum_train_epochs}_${curriculum_loss_type}"
    mkdir -p "${tune_output_dir}"
    find ../data -name 'hdsner_report_*.json' -exec mv '{}' "${tune_output_dir}" ';'
    find ../data -name 'pred_*' -exec rm '{}' ';'
done
done
)
