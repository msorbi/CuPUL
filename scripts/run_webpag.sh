python train.py --do_train --do_eval --dataset_name Webpage     \
    --train_epochs 10 --train_lr 3e-5     \
    --drop_other 0.1 --drop_entity 0.0     \
    --curriculum_train_sub_epochs 3 --curriculum_train_lr 3e-5 --curriculum_train_epochs 5     \
    --self_train_epochs 5 --self_train_lr 5e-7 --m 10 \
    --self_train_update_interval 20