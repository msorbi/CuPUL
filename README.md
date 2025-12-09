# CuPUL for Historical NER
Fork of [liyp0095/CuPUL](https://github.com/liyp0095/CuPUL) with added datasets. Work in progress.

## Historical NER Instructions

### Datasets preparation
Clone the `hdsner-utils` submodule, and follow its README to download the texts and create the environment.
```bash
source scripts/prepare_rcnum.sh
```
  - clones the datasets submodule
  - creates the datasets conda environment
  - downloads and pre-processes the datasets, with sequence length 64

### Environment setup
This is the setup of the model environment, which differs from the one in the submodule.
```bash
conda env create -n CuPUL-flair -f environment.yml
conda activate CuPUL-flair
```

### Format data and run model
```bash
bash scripts/run_rcnum.sh
```
Results will be in `data/hdsner-DATASET-(Fully|Dict_*)`. \
**NOTE**: This will overwrite previous results.
