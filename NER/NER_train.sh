set -e

python3 bert_lstm_ner.py \
      -task_name="NER" \
      -do_train=True \
      -do_eval=True \
      -do_predict=True \
      -max_seq_length=128 \
      -train_batch_size=32 \
      -learning_rate=2e-5 \
      -num_train_epochs=3
