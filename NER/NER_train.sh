set -e

python3 bert_lstm_ner.py \
    -device_map=0 \
    -do_train=True \
    -do_eval=True \
    -do_predict=False \
    -max_seq_length=128 \
    -batch_size=32 \
    -learning_rate=2e-5 \
    -num_train_epochs=15
