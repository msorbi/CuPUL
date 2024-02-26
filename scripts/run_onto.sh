python train.py --do_train --do_eval --dataset_name Ontonote_5.0 \
    --loss_type CE --m 0.6 \
    --output_dir output_ce06 --temp_dir temp_ce06 \
    --train_epochs 1 --train_lr 3e-5 \
    --drop_other 0.2 --drop_entity 0.0 \
    --curriculum_train_sub_epochs 1 --curriculum_train_lr 3e-5 --curriculum_train_epochs 5 \
    --self_train_lr 5e-7 --self_train_epochs 1 \
    --max_seq_length 230 --self_train_update_interval 500 
