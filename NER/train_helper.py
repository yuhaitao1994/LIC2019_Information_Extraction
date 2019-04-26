# -*- coding: utf-8 -*-

"""

 @Time    : 2019/1/30 14:01
 @Author  : MaCan (ma_cancan@163.com)
 @File    : train_helper.py
"""

import argparse
import os

__all__ = ['get_args_parser']


def get_args_parser():
    parser = argparse.ArgumentParser()

    bert_path = '../bert/bert_model'
    root_path = '../'

    parser.add_argument('-experiment_name', type=str,
                        help='name', required=True)
    parser.add_argument('-data_dir', type=str, default=os.path.join(root_path, 'data/NER_data'),
                        help='train, dev and test data dir')
    parser.add_argument('-bert_config_file', type=str,
                        default=os.path.join(bert_path, 'bert_config.json'))
    parser.add_argument('-output_dir', type=str, default=os.path.join(root_path, 'data', 'NER_model_' + parser.parse_args().experiment_name),
                        help='directory of a pretrained BERT model')
    parser.add_argument('-init_checkpoint', type=str, default=os.path.join(bert_path, 'bert_model.ckpt'),
                        help='Initial checkpoint (usually from a pre-trained BERT model).')
    parser.add_argument('-vocab_file', type=str, default=os.path.join(bert_path, 'vocab.txt'),
                        help='')

    parser.add_argument('-max_seq_length', type=int, default=128,
                        help='The maximum total input sequence length after WordPiece tokenization.')
    parser.add_argument('-do_train', type=bool, default=False,
                        help='Whether to run training.')
    parser.add_argument('-do_eval', type=bool, default=False,
                        help='Whether to run eval on the dev set.')
    parser.add_argument('-do_predict', type=bool, default=False,
                        help='Whether to run the predict in inference mode on the test set.')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Total batch size for training, eval and predict.')
    parser.add_argument('-learning_rate', type=float, default=2e-5,
                        help='The initial learning rate for Adam.')
    parser.add_argument('-num_train_epochs', type=float, default=15,
                        help='Total number of training epochs to perform.')
    parser.add_argument('-dropout_rate', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('-clip', type=float, default=0.5,
                        help='Gradient clip')
    parser.add_argument('-warmup_proportion', type=float, default=0.1,
                        help='Proportion of training to perform linear learning rate warmup for '
                             'E.g., 0.1 = 10% of training.')
    parser.add_argument('-lstm_size', type=int, default=128,
                        help='size of lstm units.')
    parser.add_argument('-num_layers', type=int, default=1,
                        help='number of rnn layers, default is 1.')
    parser.add_argument('-cell', type=str, default='lstm',
                        help='which rnn cell used.')
    parser.add_argument('-save_checkpoints_steps', type=int, default=500,
                        help='save_checkpoints_steps')
    parser.add_argument('-save_summary_steps', type=int, default=500,
                        help='save_summary_steps.')
    parser.add_argument('-filter_adam_var', type=bool, default=True,
                        help='after training do filter Adam params from model and save no Adam params model in file.')
    parser.add_argument('-do_lower_case', type=bool, default=True,
                        help='Whether to lower case the input text.')
    parser.add_argument('-clean', type=bool, default=True)
    parser.add_argument('-device_map', type=str, default='0',
                        help='witch device using to train')

    # add labels
    parser.add_argument('-label_list', type=str, default=None,
                        help='User define labels， can be a file with one label one line or a string using \',\' split')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-ner', type=str, default='ner',
                        help='which modle to train')

    return parser.parse_args()
