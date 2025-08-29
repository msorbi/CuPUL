#!/bin/bash
# hdsner datasets preparation
( 
    # clone datasets repository
    cd hdsner-utils

    # create and activate environment
    conda activate hdsner

    # prepare datasets
    bash src/rcnum_preprocess.sh --dictsizes 100 -- --max-seq-length 64

    # deactivate environment and return to project directory
    conda deactivate
    cd ..
)
