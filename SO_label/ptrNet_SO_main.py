# -*- coding:utf-8 -*-
"""
RC主函数
"""
import collections
import os
import numpy as np
import tensorflow as tf
import codecs
import pickle
import sys
from sklearn import metrics

sys.path.append("../")
from bert.bert_code import modeling, optimization, tokenization
from models import create_model_ptr, InputFeatures_ptr, InputExample
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()

    bert_path = '../bert/bert_model'
    root_path = '../data/'

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser.add_argument('-experiment_name', type=str, default='1',
                        help='name')
    parser.add_argument('-data_dir', type=str, default=os.path.join(root_path, 'SO_data'),
                        help='train, dev and test data dir')
    parser.add_argument('-bert_config_file', type=str,
                        default=os.path.join(bert_path, 'bert_config.json'))
    parser.add_argument('-output_dir', type=str, default=root_path,
                        help='directory of a pretrained BERT model')
    parser.add_argument('-init_checkpoint', type=str, default=os.path.join(bert_path, 'bert_model.ckpt'),
                        help='Initial checkpoint (usually from a pre-trained BERT model).')
    parser.add_argument('-vocab_file', type=str, default=os.path.join(bert_path, 'vocab.txt'),
                        help='')
    parser.add_argument('-max_seq_length', type=int, default=150,
                        help='The maximum total input sequence length after WordPiece tokenization.')
    parser.add_argument('-do_train', type=str2bool, default=False,
                        help='Whether to run training.')
    parser.add_argument('-do_eval', type=str2bool, default=False,
                        help='Whether to run eval on the dev set.')
    parser.add_argument('-do_predict', type=str2bool, default=True,
                        help='Whether to run the predict in inference mode on the test set.')
    parser.add_argument('-batch_size', type=int, default=32,
                        help='Total batch size for training, eval and predict.')
    parser.add_argument('-learning_rate', type=float, default=2e-5,
                        help='The initial learning rate for Adam.')
    parser.add_argument('-num_train_epochs', type=float, default=5,
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
    parser.add_argument('-save_checkpoints_steps', type=int, default=1000,
                        help='save_checkpoints_steps')
    parser.add_argument('-save_summary_steps', type=int, default=1000,
                        help='save_summary_steps.')
    parser.add_argument('-filter_adam_var', type=str2bool, default=False,
                        help='after training do filter Adam params from model and save no Adam params model in file.')
    parser.add_argument('-do_lower_case', type=str2bool, default=True,
                        help='Whether to lower case the input text.')
    parser.add_argument('-clean', type=str2bool, default=True)
    parser.add_argument('-device_map', type=str, default='1',
                        help='witch device using to train')

    # add labels
    parser.add_argument('-label_list', type=str, default='../dict/p_eng',
                        help='User define labels， can be a file with one label one line or a string using \',\' split')

    parser.add_argument('-verbose', action='store_true', default=False,
                        help='turn on tensorflow logging for debug')
    parser.add_argument('-ptr', type=str, default='Ptr',
                        help='which modle to train')

    return parser.parse_args()


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

    @classmethod
    def _read_data(cls, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                data = []
                context = line.strip().split('\t')
                if len(context) != 4:
                    continue
                data.append(context[0])
                data.append(context[1])
                data.append(context[2])
                data.append(context[3])
                lines.append(data)
            return lines


class PtrProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = list()
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, labels=None):
        if labels is not None:
            try:
                # 支持从文件中读取标签类型,读取的是p_eng的标签
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip().split()[-1])
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
            except Exception as e:
                print(e)
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[2] + '&&' + line[3])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    ptr Net做主客体标注
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)

    over = 0
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
        over = 1

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # 生成主客体的首尾指针和实体关系类别的id
    sub, obj = example.text_b.split('&&')
    sub = tokenizer.tokenize(sub)
    obj = tokenizer.tokenize(obj)

    sub_head = -1
    sub_tail = -1
    obj_head = -1
    obj_tail = -1

    for i in range(len(tokens) - 1):
        cut = tokens[i:min(i + len(sub), len(tokens) - 1)]
        if ''.join(''.join(cut).split('##')) == ''.join(''.join(sub).split('##')):
            sub_head = i
            sub_tail = i + len(cut) - 1
            break
        cut = tokens[i:min(i + len(sub) - 1, len(tokens) - 1)]
        if ''.join(''.join(cut).split('##')) == ''.join(''.join(sub).split('##')):
            sub_head = i
            sub_tail = i + len(cut) - 1
            break
        cut = tokens[i:min(i + len(sub) + 1, len(tokens) - 1)]
        if ''.join(''.join(cut).split('##')) == ''.join(''.join(sub).split('##')):
            sub_head = i
            sub_tail = i + len(cut) - 1
            break
    if sub_head == -1:
        sub_head = sub_tail = len(tokens) - 1
        # print(tokens)
        # print(sub)
        # raise ValueError

    for i in range(len(tokens) - 1):
        cut = tokens[i:min(i + len(obj), len(tokens) - 1)]
        if ''.join(''.join(cut).split('##')) == ''.join(''.join(obj).split('##')):
            obj_head = i
            obj_tail = i + len(cut) - 1
            break
        cut = tokens[i:min(i + len(obj) - 1, len(tokens) - 1)]
        if ''.join(''.join(cut).split('##')) == ''.join(''.join(obj).split('##')):
            obj_head = i
            obj_tail = i + len(cut) - 1
            break
        cut = tokens[i:min(i + len(obj) + 1, len(tokens) - 1)]
        if ''.join(''.join(cut).split('##')) == ''.join(''.join(obj).split('##')):
            obj_head = i
            obj_tail = i + len(cut) - 1
            break
    if obj_head == -1:
        obj_head = obj_tail = len(tokens) - 1

    label_id = label_map[example.label]

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
        tf.logging.info("pointer: %d %d %d %d" %
                        (sub_head, sub_tail, obj_head, obj_tail))

    feature = InputFeatures_ptr(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        sub_ptr=[sub_head, sub_tail],
        obj_ptr=[obj_head, obj_tail])
    return feature, over


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """
    将数据转化为TF_Record 结构，作为模型数据输入
    :param examples:  样本
    :param label_list:标签list
    :param max_seq_length: 预先设定的最大序列长度
    :param tokenizer: tokenizer 对象
    :param output_file: tf.record 输出路径
    :param mode:
    :return:
    """
    writer = tf.python_io.TFRecordWriter(output_file)
    Over = 0
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature, over = convert_single_example(
            ex_index, example, label_list, max_seq_length, tokenizer)
        Over += over

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        features["sub_ptr"] = create_int_feature(feature.sub_ptr)
        features["obj_ptr"] = create_int_feature(feature.obj_ptr)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    print("over length:{:.5f}".format(Over / ex_index))


def file_based_dataset(input_file, batch_size, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "sub_ptr": tf.FixedLenFeature([2], tf.int64),
        "obj_ptr": tf.FixedLenFeature([2], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    d = tf.data.TFRecordDataset(input_file)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=10000)
    d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                   batch_size=batch_size,
                                                   num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                   drop_remainder=drop_remainder))
    d = d.prefetch(buffer_size=4)
    return d


def get_last_checkpoint(model_path):
    if not os.path.exists(os.path.join(model_path, 'checkpoint')):
        tf.logging.info('checkpoint file not exits:'.format(
            os.path.join(model_path, 'checkpoint')))
        return None
    last = None
    with codecs.open(os.path.join(model_path, 'checkpoint'), 'r', encoding='utf-8') as fd:
        for line in fd:
            line = line.strip().split(':')
            if len(line) != 2:
                continue
            if line[0] == 'model_checkpoint_path':
                last = line[1][2:-1]
                break
    return last


def adam_filter(model_path):
    """
    去掉模型中的Adam相关参数，这些参数在测试的时候是没有用的
    :param model_path: 
    :return: 
    """
    last_name = get_last_checkpoint(model_path)
    if last_name is None:
        return
    with tf.Session(graph=tf.Graph()) as sess:
        imported_meta = tf.train.import_meta_graph(
            os.path.join(model_path, last_name + '.meta'))
        imported_meta.restore(sess, os.path.join(model_path, last_name))
        need_vars = []
        for var in tf.global_variables():
            if 'adam_v' not in var.name and 'adam_m' not in var.name:
                need_vars.append(var)
        saver = tf.train.Saver(need_vars)
        saver.save(sess, os.path.join(model_path, 'model.ckpt'))


def result_to_pair(label_list, writer, data_file, result):
    f = open(data_file, 'r')
    for line, prediction in zip(f, result):
        line = line.strip()
        label = label_list[prediction]
        writer.write(line + '\t' + str(label) + '\n')


def train_and_eval(args, processor, tokenizer, bert_config, sess_config, label_list):
    """
    训练和评估函数
    """

    # 生成tf_record文件
    train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_dev_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) * 1.0 / args.batch_size * args.num_train_epochs)
    if num_train_steps < 1:
        raise AttributeError('training data is so small...')
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", args.batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", len(eval_examples))
    tf.logging.info("  Batch size = %d", args.batch_size)

    # 写入tfrecord
    train_file = os.path.join(args.output_dir, "train.tf_record")
    if not os.path.exists(train_file):
        filed_based_convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, train_file)
    eval_file = os.path.join(args.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
        filed_based_convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, eval_file)

    """
    -------------分割线-------------
    """
    # 存储路径
    log_dir = os.path.join(args.output_dir, 'log')
    save_dir = os.path.join(args.output_dir, 'model')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # # 加载数据
    # train_file = os.path.join(args.output_dir, "train.tf_record")
    # eval_file = os.path.join(args.output_dir, "eval.tf_record")
    # if not os.path.exists(train_file) or not os.path.exists(eval_file):
    #     raise ValueError
    # 生成dataset
    train_data = file_based_dataset(input_file=train_file, batch_size=args.batch_size,
                                    seq_length=args.max_seq_length, is_training=True, drop_remainder=False)
    eval_data = file_based_dataset(input_file=eval_file, batch_size=args.batch_size,
                                   seq_length=args.max_seq_length, is_training=False, drop_remainder=False)
    train_iter = train_data.make_one_shot_iterator().get_next()

    # 开启计算图
    with tf.Session(config=sess_config) as sess:
        # 构造模型
        input_ids = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32, name='input_ids')
        input_mask = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32, name='input_mask')
        segment_ids = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32, name='segment_ids')
        label_ids = tf.placeholder(
            shape=[None], dtype=tf.int32, name='label_ids')
        is_training = tf.get_variable(
            "is_training", shape=[], dtype=tf.bool, trainable=False)

        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids, len(label_list))
        pred_ids = tf.argmax(probabilities, axis=-1,
                             output_type=tf.int32, name="pred_ids")

        # 优化器
        train_op = optimization.create_optimizer(
            total_loss, args.learning_rate, num_train_steps, num_warmup_steps, False)
        sess.run(tf.global_variables_initializer())

        # 加载bert原始模型
        tvars = tf.trainable_variables()
        if args.init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(
                    tvars, args.init_checkpoint)
            tf.train.init_from_checkpoint(args.init_checkpoint, assignment_map)

        # 打印加载模型的参数
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        # 初始化存储和log
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        saver = tf.train.Saver()

        # 定义一些全局变量
        best_eval_acc = 0.0
        patience = 0

        # 开始训练
        sess.run(tf.assign(is_training, tf.constant(True, dtype=tf.bool)))
        for go in range(1, num_train_steps + 1):
            # feed
            train_batch = sess.run(train_iter)
            loss, preds, op = sess.run([total_loss, pred_ids, train_op], feed_dict={
                input_ids: train_batch['input_ids'], input_mask: train_batch['input_mask'],
                segment_ids: train_batch['segment_ids'], label_ids: train_batch['label_ids']})

            if go % args.save_summary_steps == 0:
                # 训练log
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="loss/train_loss", simple_value=loss / args.batch_size), ]), sess.run(tf.train.get_global_step()))
                writer.flush()

            if go % args.save_checkpoints_steps == 0:
                # 验证集评估
                sess.run(tf.assign(is_training, tf.constant(False, dtype=tf.bool)))
                eval_loss_total = 0.0
                eval_preds_total = np.array([0], dtype=np.int32)
                eval_truth_total = np.array([0], dtype=np.int32)
                # 重新生成一次验证集数据
                eval_data = eval_data.repeat()
                eval_iter = eval_data.make_one_shot_iterator().get_next()
                # for _ in range(0, int(len(eval_examples) / args.batch_size) + 1):
                # eval集太大，这样每次用全部的话太耗费时间
                for _ in range(1000):
                    # eval feed
                    eval_batch = sess.run(eval_iter)
                    eval_loss, eval_preds, eval_truth = sess.run([total_loss, pred_ids, label_ids], feed_dict={
                        input_ids: eval_batch['input_ids'], input_mask: eval_batch['input_mask'],
                        segment_ids: eval_batch['segment_ids'], label_ids: eval_batch['label_ids']})
                    # 统计结果
                    eval_loss_total += eval_loss
                    eval_preds_total = np.concatenate(
                        (eval_preds_total, eval_preds))
                    eval_truth_total = np.concatenate(
                        (eval_truth_total, eval_truth))

                # 处理评估结果，计算recall与f1
                eval_preds_total = eval_preds_total[1:]
                eval_truth_total = eval_truth_total[1:]
                eval_f1 = metrics.f1_score(
                    eval_truth_total, eval_preds_total, average='macro')
                eval_acc = metrics.accuracy_score(
                    eval_truth_total, eval_preds_total)
                eval_loss_aver = eval_loss_total / 1000

                # 评估实体关系分类的指标

                # 评估log
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="loss/eval_loss", simple_value=eval_loss_aver), ]), sess.run(tf.train.get_global_step()))
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="eval/f1", simple_value=eval_f1), ]), sess.run(tf.train.get_global_step()))
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="eval/acc", simple_value=eval_acc), ]), sess.run(tf.train.get_global_step()))
                writer.flush()

                # early stopping 与 模型保存
                if eval_acc <= best_eval_acc:
                    patience += 1
                    if patience >= 100:
                        print("early stoping!")
                        return

                if eval_acc > best_eval_acc:
                    patience = 0
                    best_eval_acc = eval_acc
                    saver.save(sess, os.path.join(save_dir, "model_{}_acc_{:.4f}.ckpt".format(
                        sess.run(tf.train.get_global_step()), best_eval_acc)))

                sess.run(tf.assign(is_training, tf.constant(False, dtype=tf.bool)))


def predict(args, processor, tokenizer, bert_config, sess_config, label_list):
    """
    预测函数
    """
    # 生成3个examples
    predict_examples = processor.get_test_examples(args.data_dir)
    predict_file = os.path.join(args.output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(
        predict_examples, label_list, args.max_seq_length, tokenizer, predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d", len(predict_examples))
    tf.logging.info("  Batch size = %d", args.batch_size)
    train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_dev_examples(args.data_dir)
    train_file = os.path.join(args.output_dir, "train.tf_record")
    eval_file = os.path.join(args.output_dir, "eval.tf_record")
    # 生成数据集
    train_data = file_based_dataset(input_file=train_file, batch_size=args.batch_size,
                                    seq_length=args.max_seq_length, is_training=False, drop_remainder=False)
    eval_data = file_based_dataset(input_file=eval_file, batch_size=args.batch_size,
                                   seq_length=args.max_seq_length, is_training=False, drop_remainder=False)
    predict_data = file_based_dataset(input_file=predict_file, batch_size=args.batch_size,
                                      seq_length=args.max_seq_length, is_training=False, drop_remainder=False)
    train_iter = train_data.make_one_shot_iterator().get_next()
    eval_iter = eval_data.make_one_shot_iterator().get_next()
    predict_iter = predict_data.make_one_shot_iterator().get_next()

    # 开启计算图
    with tf.Session(config=sess_config) as sess:
        # 从文件中读取计算图
        save_dir = os.path.join(args.output_dir, 'model')
        saver = tf.train.import_meta_graph(
            tf.train.latest_checkpoint(save_dir) + ".meta")
        sess.run(tf.global_variables_initializer())
        # 打印张量名
        # tensor_list = [
        #     n.name for n in tf.get_default_graph().as_graph_def().node if 'older' in n.name]
        # print(tensor_list)
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        # 通过张量名获取模型的占位符和参数
        input_ids = tf.get_default_graph().get_tensor_by_name('input_ids:0')
        input_mask = tf.get_default_graph().get_tensor_by_name('input_mask:0')
        segment_ids = tf.get_default_graph().get_tensor_by_name('segment_ids:0')
        label_ids = tf.get_default_graph().get_tensor_by_name('label_ids:0')
        sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(
            'is_training:0'), tf.constant(False, dtype=tf.bool)))
        # 找到crf输出, 注意其名称在crf_decode源码中, 可以在graph中查到
        pred_ids = tf.get_default_graph().get_tensor_by_name('pred_ids:0')

        # test集预测
        predict_total = np.array([0] * 128, dtype=np.int32)
        for _ in range(0, int(len(predict_examples) / args.batch_size) + 1):
            # predict feed
            predict_batch = sess.run(predict_iter)
            predict_res = sess.run(pred_ids, feed_dict={
                input_ids: predict_batch['input_ids'], input_mask: predict_batch['input_mask'],
                segment_ids: predict_batch['segment_ids'], label_ids: predict_batch['label_ids']})
            predict_total = np.concatenate((predict_total, predict_res))
        # 处理评估结果，计算recall与f1
        predict_total = predict_total[1:]
        output_predict_file = os.path.join(
            args.output_dir, "prediction_test.txt")
        with codecs.open(output_predict_file, 'w', encoding='utf-8') as writer:
            result_to_pair(label_list, writer, os.path.join(
                args.data_dir, 'test.txt'), predict_total)

        # eval集预测
        eval_total = np.array([0], dtype=np.int32)
        for _ in range(0, int(len(eval_examples) / args.batch_size) + 1):
            # predict feed
            eval_batch = sess.run(eval_iter)
            eval_res = sess.run(pred_ids, feed_dict={
                input_ids: eval_batch['input_ids'], input_mask: eval_batch['input_mask'],
                segment_ids: eval_batch['segment_ids'], label_ids: eval_batch['label_ids']})
            eval_total = np.concatenate((eval_total, eval_res))
        # 处理评估结果，计算recall与f1
        eval_total = eval_total[1:]
        output_eval_file = os.path.join(args.output_dir, "prediction_dev.txt")
        with codecs.open(output_eval_file, 'w', encoding='utf-8') as writer:
            result_to_pair(label_list, writer, os.path.join(
                args.data_dir, 'dev.txt'), eval_total)


if __name__ == '__main__':
    """
    开始执行
    """
    args = get_args_parser()
    args.output_dir = os.path.join(
        args.output_dir, 'SO_model_' + args.experiment_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    tf.logging.set_verbosity(tf.logging.INFO)
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v)
                               for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' %
              (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

    processors = {
        "Ptr": PtrProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # 在re train 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        print('hahaha')
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    # check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # 创建ptr dataprocessor对象
    processor = processors[args.ptr](args.output_dir)
    label_list = processor.get_labels(labels=args.label_list)
    print(len(label_list))

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    if args.do_train and args.do_eval:
        train_and_eval(args=args, processor=processor, tokenizer=tokenizer,
                       bert_config=bert_config, sess_config=session_config, label_list=label_list)
        # if args.filter_adam_var:
        #     adam_filter(os.path.join(args.output_dir, 'model'))

    if args.do_predict:
        predict(args=args, processor=processor, tokenizer=tokenizer,
                bert_config=bert_config, sess_config=session_config, label_list=label_list)
