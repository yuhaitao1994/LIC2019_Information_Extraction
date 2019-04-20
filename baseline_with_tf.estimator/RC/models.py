# -*- coding: utf-8 -*-

"""
 一些公共模型代码
 @Time    : 2019/1/30 12:46
 @Author  : MaCan (ma_cancan@163.com)
 @File    : models.py
"""

import sys
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
sys.path.append("../")
from bert.bert_code import modeling, optimization, tokenization

__all__ = ['InputExample', 'InputFeatures', 'decode_labels', 'create_model', 'convert_id_str',
           'convert_id_to_label', 'result_to_json', 'create_classification_model']


class Model(object):
    def __init__(self, *args, **kwargs):
        pass


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids, labels, num_labels):
    """

    :param bert_config:
    :param is_training:
    :param input_ids:
    :param input_mask:
    :param segment_ids:
    :param labels:
    :param num_labels:
    :param use_one_hot_embedding:
    :return:
    """
    # 通过传入的训练数据，进行representation
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def decode_labels(labels, batch_size):
    new_labels = []
    for row in range(batch_size):
        label = []
        for i in labels[row]:
            i = i.decode('utf-8')
            if i == '**PAD**':
                break
            if i in ['[CLS]', '[SEP]']:
                continue
            label.append(i)
        new_labels.append(label)
    return new_labels


def convert_id_str(input_ids, batch_size):
    res = []
    for row in range(batch_size):
        line = []
        for i in input_ids[row]:
            i = i.decode('utf-8')
            if i == '**PAD**':
                break
            if i in ['[CLS]', '[SEP]']:
                continue

            line.append(i)
        res.append(line)
    return res


def convert_id_to_label(pred_ids_result, idx2label, batch_size):
    """
    将id形式的结果转化为真实序列结果
    :param pred_ids_result:
    :param idx2label:
    :return:
    """
    result = []
    index_result = []
    for row in range(batch_size):
        curr_seq = []
        curr_idx = []
        ids = pred_ids_result[row]
        for idx, id in enumerate(ids):
            if id == 0:
                break
            curr_label = idx2label[id]
            if curr_label in ['[CLS]', '[SEP]']:
                if id == 102 and (idx < len(ids) and ids[idx + 1] == 0):
                    break
                continue
            # elif curr_label == '[SEP]':
            #     break
            curr_seq.append(curr_label)
            curr_idx.append(id)
        result.append(curr_seq)
        index_result.append(curr_idx)
    return result, index_result


def result_to_json(self, string, tags):
    """
    将模型标注序列和输入序列结合 转化为结果
    :param string: 输入序列
    :param tags: 标注结果
    :return:
    """
    item = {"entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    last_tag = ''

    for char, tag in zip(string, tags):
        if tag[0] == "S":
            self.append(char, idx, idx + 1, tag[2:])
            item["entities"].append(
                {"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
        elif tag[0] == "B":
            if entity_name != '':
                self.append(entity_name, entity_start, idx, last_tag[2:])
                item["entities"].append(
                    {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                entity_name = ""
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "O":
            if entity_name != '':
                self.append(entity_name, entity_start, idx, last_tag[2:])
                item["entities"].append(
                    {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
                entity_name = ""
        else:
            entity_name = ""
        entity_start = idx
        idx += 1
        last_tag = tag
    if entity_name != '':
        self.append(entity_name, entity_start, idx, last_tag[2:])
        item["entities"].append(
            {"word": entity_name, "start": entity_start, "end": idx, "type": last_tag[2:]})
    return item
