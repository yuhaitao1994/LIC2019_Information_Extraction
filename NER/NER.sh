# -*- coding:utf-8 -*-

python3 ner_main.py \
    -device_map=0 \
    -do_train=True \
    -do_eval=True \
    -do_predict=False \
    -max_seq_length=128 \
    -batch_size=32 \
    -learning_rate=2e-5 \
    -num_train_epochs=15 \
    -save_summary_steps=500 \
    -save_checkpoints_steps=500 \
    -filter_adam_var=True
