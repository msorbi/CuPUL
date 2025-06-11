#!/bin/bash

(
# set attributes
dataset_prefix="data/hdsner-"
nl=$'\n'

datasets="`echo ${dataset_prefix}*`"

# move to directory and activate evaluation environment
cd hdsner-utils/
conda activate hdsner

# execute on all datasets
for split in valid test
do
    for stage in ct st
    do
        for dataset in ${datasets}
        do
            if [ -d "../${dataset}" ]
            then
                output_file="../${dataset}/pred_${split}_${stage}.json"
                python3 src/eval.py \
                    --true "../${dataset}/${split}.txt" \
                    --pred <(cut "../${dataset}/pred_${split}_${stage}.txt" -d ' ' -f 1,2) \
                    --output "$output_file" \
                    --n 1 \
                    --field-delimiter ' ' \
                > /dev/null
                echo "$output_file" # this is going to python below
            fi
        done | \
python3 -c "import sys ${nl}\
import json ${nl}\
summary = {} ${nl}\
for f in sys.stdin: ${nl}\
    with open(f.strip(), 'r') as fp: ${nl}\
        x = json.load(fp) ${nl}\
    summary[f.strip().split('/')[-2]] = x ${nl}\
with open(\"../data/hdsner_report_${split}_${stage}.json\", 'w') as fp: ${nl}\
    json.dump(obj=summary, fp=fp) ${nl}\
"
    done
done

# deactivate environment and return to project directory
conda deactivate
cd ..

)
