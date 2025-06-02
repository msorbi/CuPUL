python train.py --do_train --do_eval --dataset_name CoNLL2003_KB \
    --train_epochs 1 --train_lr 2e-5 \
    --drop_other 0.5 --drop_entity 0.0 \
    --curriculum_train_sub_epochs 1 --curriculum_train_lr 2e-5 --curriculum_train_epochs 5 \
    --self_train_lr 5e-7 --self_train_epochs 5 --m 20