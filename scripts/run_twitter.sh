python train.py --do_train --do_eval --dataset_name Twitter \
    --train_epochs 5 --train_lr 1e-5 \
    --drop_other 0.1 --drop_entity 0.1 \
    --curriculum_train_sub_epochs 3 --curriculum_train_lr 2e-5 --curriculum_train_epochs 5 \
    --self_train_lr 5e-6 --self_train_epochs 5  \
    --self_train_update_interval 150 --entity_threshold 0.9