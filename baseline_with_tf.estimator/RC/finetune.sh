#!/usr/bin/env bash
dir=/home/jinzhiwei/data/classification
CUDA_VISIBLE_DEVICES=0 python run_classifier_test.py \
    --do_train=true \
    --do_eval=true \
    --input_files=$dir/data/20181218/train.tf \
    --eval_file=$dir/data/20181218/test.tf \
    --num_labels=186 \
    --vocab_file=$dir/dl/vocab_new.txt \
    --bert_config_file=$dir/dl/bert_config.json \
    --init_checkpoint=/mnt/mmu/jinzhiwei/bert/data/model_newdict/model.ckpt-100000 \
    --train_batch_size=64 \
    --eval_batch_size=64 \
    --max_seq_length=80 \
    --num_train_steps=80000 \
    --learning_rate=1e-4 \
    --output_dir=$dir/dl/ckpt1218/
