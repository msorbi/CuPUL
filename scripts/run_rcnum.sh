cd src

dataset_prefix="../data/rcnum"
rm -r ${dataset_prefix}*
for setting in `ls ../hdsner-utils/data/`
do
    source="../hdsner-utils/data/${setting}/rcnum/"
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
        --do_train --dataset_name "`echo "${dataset}" | cut -d '/' -f 3`" \
        --train_epochs 15 --train_lr 1e-3 \
        --drop_other 0.1 --drop_entity 0.0 \
        --loss_type MPN \
        --curriculum_train_sub_epochs 1 --curriculum_train_lr 1e-3 --curriculum_train_epochs 20 \
        --curriculum_loss_type MPN \
        --self_train_epochs 0 \
        --m 20 \
        --no_gt_output \
        --do_eval --eval_on valid \
    > "${dataset}/stdout.txt" 2> "${dataset}/stderr.txt"
done

# copy predictions to utils directory
cd ..
cp data/rcnum-IOB-Fully/pred_test_ct.txt hdsner-utils/data/data_raw/rcnum/pred.iob
