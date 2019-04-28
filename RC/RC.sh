# -*- coding:utf-8 -*-

python3 rc_main.py \
    -device_map=1 \
    -do_train=True \
    -do_eval=True \
    -do_predict=True \
    -max_seq_length=150 \
    -batch_size=32 \
    -learning_rate=2e-5 \
    -num_train_epochs=5 \
    -save_summary_steps=5 \
    -save_checkpoints_steps=5 \
    -filter_adam_var=False \
    -experiment_name=demo
