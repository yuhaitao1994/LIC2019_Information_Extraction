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


class InputFeatures_ptr(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_ptr, obj_ptr, ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.sub_ptr = sub_ptr
        self.obj_ptr = obj_ptr


def relation_embedding(relation, num_relations, dim):
    """
    relation embedding
    """
    with tf.variable_scope("relation_embedding"):
        relation_embedding = tf.get_variable('relation_matrix', shape=[num_relations, dim],
                                             dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
        relation_output = tf.nn.embedding_lookup(
            relation_embedding, relation)
    return relation_output


def softmax_mask(val, mask):
    INF = 1e30
    return -INF * (1 - tf.cast(mask, tf.float32)) + val


def dense(inputs, hidden, use_bias=True, scope="dense"):
    """
    全连接层
    """
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        # 三维的inputs，reshape成二维
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        # outshape就是input的最后一维变成hidden
        res = tf.reshape(res, out_shape)
        return res


class ptr_net:
    def __init__(self, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = tf.contrib.rnn.GRUCell(hidden)
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train

    def __call__(self, init, match, hidden, mask):
        with tf.variable_scope(self.scope):
            d_match = tf.cond(self.is_train, lambda: tf.nn.dropout(
                match, keep_prob=self.keep_prob), lambda: match)
            inp, logits1 = pointer(d_match, init, hidden, mask)
            d_inp = tf.cond(self.is_train, lambda: tf.nn.dropout(
                inp, keep_prob=self.keep_prob), lambda: inp)
            tf.get_variable_scope().reuse_variables()
            ptr_out, logits2 = pointer(d_match, d_inp, hidden, mask)
            return logits1, logits2, ptr_out


def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        res = dense(res, 128, use_bias=False, scope="res")
        return res, s1


def create_model_ptr(bert_config, is_training, input_ids, input_mask, segment_ids, labels, sub_ptr, obj_ptr, num_labels):
    """
    SO labeling, 基于ptr Net
    """
    # 首先使用bert的输出作为embedding
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
    )
    bert_out = model.get_sequence_output()

    bert_out = tf.cond(is_training, lambda: tf.nn.dropout(
        bert_out, keep_prob=0.9), lambda: bert_out)

    # relation embedding 和 pointer network 的流程
    relation_init = relation_embedding(labels, num_labels, 128)

    # 两个pointer network，分别指向主客体，参数不重用，但是第一个的结果作为第二个的init
    sub_ptr_net = ptr_net(hidden=512, keep_prob=0.9,
                          is_train=is_training, scope='sub_ptrNet')
    obj_ptr_net = ptr_net(hidden=512, keep_prob=0.9,
                          is_train=is_training, scope='obj_ptrNet')

    sub_logits1, sub_logits2, sub_state = sub_ptr_net(
        relation_init, bert_out, 512, input_mask)
    obj_logits1, obj_logits2, _ = obj_ptr_net(
        sub_state, bert_out, 512, input_mask)

    with tf.variable_scope('loss'):
        sub_outer = tf.matmul(tf.expand_dims(tf.nn.softmax(
            sub_logits1), axis=2), tf.expand_dims(tf.nn.softmax(sub_logits2), axis=1))
        sub_outer = tf.matrix_band_part(sub_outer, 0, 15)
        sub_h_preds = tf.argmax(tf.reduce_max(sub_outer, axis=2), axis=1)
        sub_t_preds = tf.argmax(tf.reduce_max(sub_outer, axis=1), axis=1)

        obj_outer = tf.matmul(tf.expand_dims(tf.nn.softmax(
            obj_logits1), axis=2), tf.expand_dims(tf.nn.softmax(obj_logits2), axis=1))
        obj_outer = tf.matrix_band_part(obj_outer, 0, 15)
        obj_h_preds = tf.argmax(tf.reduce_max(obj_outer, axis=2), axis=1)
        obj_t_preds = tf.argmax(tf.reduce_max(obj_outer, axis=1), axis=1)

        loss_s_h = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=sub_logits1, labels=tf.stop_gradient(tf.one_hot(sub_ptr[:, :1], depth=150, dtype=tf.float32)))
        loss_s_t = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=sub_logits2, labels=tf.stop_gradient(tf.one_hot(sub_ptr[:, 1:], depth=150, dtype=tf.float32)))
        loss_o_h = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=obj_logits1, labels=tf.stop_gradient(tf.one_hot(obj_ptr[:, :1], depth=150, dtype=tf.float32)))
        loss_o_t = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=obj_logits2, labels=tf.stop_gradient(tf.one_hot(obj_ptr[:, 1:], depth=150, dtype=tf.float32)))

        loss = tf.reduce_mean(loss_s_h + loss_s_t + loss_o_h + loss_o_t)

        preds = tf.concat([tf.expand_dims(sub_h_preds, axis=1), tf.expand_dims(sub_t_preds, axis=1), tf.expand_dims(
            obj_h_preds, axis=1), tf.expand_dims(obj_t_preds, axis=1)], axis=-1, name='pred_ids')

        return (loss, preds)


def create_model_SO():
    """
    主客体标注模型，使用bert + relation embedding + CRF
    """
    pass
