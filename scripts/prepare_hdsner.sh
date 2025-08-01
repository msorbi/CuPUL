#!/bin/bash
# hdsner datasets preparation
( 
    # clone datasets repository
    git submodule update --init --recursive
    cd hdsner-utils

    # create and activate environment
    conda env create -n hdsner -f environment.yml
    conda activate hdsner

    # prepare datasets
    bash src/download.sh
    bash src/preprocess.sh --dictsizes 10 20 40 60 80 100 -- --max-seq-length 64

    # deactivate environment and return to project directory
    conda deactivate
    cd ..
)
