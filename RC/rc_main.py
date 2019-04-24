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

import tf_metrics
sys.path.append("../")
from bert.bert_code import modeling, optimization, tokenization
from models import create_model, InputFeatures, InputExample
from train_helper import get_args_parser


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
                data.append(context[3])
                data.append(context[0])
                data.append(context[1] + context[2])
                lines.append(data)
            return lines


class RCProcessor(DataProcessor):
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
            text_a = tokenization.convert_to_unicode(line[1])
            text_b = tokenization.convert_to_unicode(line[2])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

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

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id)
    return feature


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
    # 遍历训练数据
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))
        # 对于每一个训练样本,
        feature = convert_single_example(
            ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature([feature.label_id])
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_dataset(input_file, batch_size, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
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
            train_examples, label_list, args.max_seq_length, tokenizer, train_file, args.output_dir)
    eval_file = os.path.join(args.output_dir, "eval.tf_record")
    if not os.path.exists(eval_file):
        filed_based_convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, eval_file, args.output_dir)

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
            shape=[None, args.max_seq_length], dtype=tf.int32)
        input_mask = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32)
        segment_ids = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32)
        label_ids = tf.placeholder(shape=[None], dtype=tf.int32)
        is_training = tf.get_variable(
            "is_training", shape=[], dtype=tf.bool, trainable=False)

        total_loss, per_example_loss, logits, probabilities = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            len(label_list), False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)
        pred_ids = tf.argmax(probabilities, axis=-1, output_type=tf.int32)

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
        best_eval_recall = 0.0
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
                eval_preds_total = np.array([[0] * 128], dtype=np.int32)
                eval_truth_total = np.array([[0] * 128], dtype=np.int32)
                # 重新生成一次验证集数据
                eval_data = eval_data.repeat()
                eval_iter = eval_data.make_one_shot_iterator().get_next()
                for _ in range(0, int(len(eval_examples) / args.batch_size) + 1):
                    # eval feed
                    eval_batch = sess.run(eval_iter)
                    eval_loss, eval_preds, eval_truth = sess.run([total_loss, pred_ids, label_ids], feed_dict={
                        input_ids: eval_batch['input_ids'], input_mask: eval_batch['input_mask'],
                        segment_ids: eval_batch['segment_ids'], label_ids: eval_batch['label_ids']})
                    # 统计结果
                    eval_loss_total += eval_loss
                    eval_preds_total = np.concatenate(
                        (eval_preds_total, eval_preds), axis=0)
                    eval_truth_total = np.concatenate(
                        (eval_truth_total, eval_truth), axis=0)

                # 处理评估结果，计算recall与f1
                eval_preds_total = eval_preds_total[1:]
                eval_truth_total = eval_truth_total[1:]
                eval_recall = metrics.recall_score(
                    eval_truth_total.reshape(-1), eval_preds_total.reshape(-1), average='macro')
                eval_f1 = metrics.f1_score(
                    eval_truth_total.reshape(-1), eval_preds_total.reshape(-1), average='macro')

                # 评估log
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="loss/eval_loss", simple_value=eval_loss_total / len(eval_examples)), ]), sess.run(tf.train.get_global_step()))
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="eval/recall", simple_value=eval_recall), ]), sess.run(tf.train.get_global_step()))
                writer.add_summary(tf.Summary(value=[tf.Summary.Value(
                    tag="eval/f1", simple_value=eval_f1), ]), sess.run(tf.train.get_global_step()))
                writer.flush()

                # early stopping 与 模型保存
                if eval_recall <= best_eval_recall:
                    patience += 1
                    if patience >= 3:
                        print("early stoping!")
                        return

                if eval_recall > best_eval_recall:
                    patience = 0
                    best_eval_recall = eval_recall
                    saver.save(sess, os.path.join(save_dir, "model_{}_recall_{:.4f}.ckpt".format(
                        sess.run(tf.train.get_global_step()), best_eval_recall)))

                sess.run(tf.assign(is_training, tf.constant(False, dtype=tf.bool)))


def predict(args):
    """
    预测函数
    """


if __name__ == '__main__':
    """
    开始执行
    """
    args = get_args_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map
    tf.logging.set_verbosity(tf.logging.INFO)
    if True:
        param_str = '\n'.join(['%20s = %s' % (k, v)
                               for k, v in sorted(vars(args).items())])
        print('usage: %s\n%20s   %s\n%s\n%s\n' %
              (' '.join(sys.argv), 'ARG', 'VALUE', '_' * 50, param_str))

    processors = {
        "RC": RCProcessor
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
    # 创建ner dataprocessor对象
    processor = processors[args.ner](args.output_dir)
    label_list = label_list = processor.get_labels()

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
        predict(args=args)
