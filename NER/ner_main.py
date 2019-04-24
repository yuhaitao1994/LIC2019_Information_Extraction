# -*- coding:utf-8 -*-
"""
主函数
"""
import collections
import os
import numpy as np
import tensorflow as tf
import codecs
import pickle
import sys
from sklearn import metrics

import tf_metrics
import conlleval
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
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[1])
                else:
                    if len(contends) == 0:
                        l = ' '.join(
                            [label for label in labels if len(label) > 0])
                        w = ' '.join([word for word in words if len(word) > 0])
                        lines.append([l, w])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    words.append('')
                    continue
            return lines


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.labels = set()
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
                # 支持从文件中读取标签类型
                if os.path.exists(labels) and os.path.isfile(labels):
                    with codecs.open(labels, 'r', encoding='utf-8') as fd:
                        for line in fd:
                            self.labels.append(line.strip())
                else:
                    # 否则通过传入的参数，按照逗号分割
                    self.labels = labels.split(',')
                self.labels = set(self.labels)  # to set
            except Exception as e:
                print(e)
        # 通过读取train文件获取标签的方法会出现一定的风险。
        if os.path.exists(os.path.join(self.output_dir, 'label_list.pkl')):
            with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'rb') as rf:
                self.labels = pickle.load(rf)
        else:
            if len(self.labels) > 0:
                self.labels = self.labels.union(set(["X", "[CLS]", "[SEP]"]))
                with codecs.open(os.path.join(self.output_dir, 'label_list.pkl'), 'wb') as rf:
                    pickle.dump(self.labels, rf)
            else:
                # 注意，这里更新了label，只有O,B,I,E四种label
                self.labels = ["O", 'B', 'I', "E", "X", "[CLS]", "[SEP]"]
        return self.labels

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            # if i == 0:
            #     print('label: ', label)
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def _read_data(self, input_file):
        """Reads a BIO data."""
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            words = []
            labels = []
            for line in f:
                contends = line.strip()
                tokens = contends.split(' ')
                if len(tokens) == 2:
                    words.append(tokens[0])
                    labels.append(tokens[-1])
                else:
                    if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
                        continue
                if contends.startswith("-DOCSTART-"):
                    continue
            return lines


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode):
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
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i
    # 保存label->index 的map
    if not os.path.exists(os.path.join(output_dir, 'label2id.pkl')):
        with codecs.open(os.path.join(output_dir, 'label2id.pkl'), 'wb') as w:
            pickle.dump(label_map, w)

    textlist = example.text.split(' ')
    labellist = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:  # 一般不会出现else
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")  # 句子开始设置CLS 标志
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")  # 句尾添加[SEP] 标志
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(
        ntokens)  # 将序列中的字(ntokens)转化为ID形式
    input_mask = [1] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        # label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    # assert len(label_mask) == max_seq_length

    # 打印部分样本数据信息
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
        tf.logging.info("label_ids: %s" %
                        " ".join([str(x) for x in label_ids]))
        # tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    # 结构化为一个类
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        # label_mask = label_mask
    )
    return feature


def filed_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file, output_dir, mode=None):
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
            ex_index, example, label_list, max_seq_length, tokenizer, output_dir, mode)

        def create_int_feature(values):
            f = tf.train.Feature(
                int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        # features["label_mask"] = create_int_feature(feature.label_mask)
        # tf.train.Example/Feature 是一种协议，方便序列化？？？
        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_dataset(input_file, batch_size, seq_length, is_training, drop_remainder):
    """
    仿照bert的file_based_input_fn_builder修改
    """
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
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
    sess = tf.Session()
    imported_meta = tf.train.import_meta_graph(
        os.path.join(model_path, last_name + '.meta'))
    imported_meta.restore(sess, os.path.join(model_path, last_name))
    need_vars = []
    for var in tf.global_variables():
        if 'adam_v' not in var.name and 'adam_m' not in var.name:
            need_vars.append(var)
    saver = tf.train.Saver(need_vars)
    saver.save(sess, os.path.join(model_path, 'model.ckpt'))


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
            shape=[None, args.max_seq_length], dtype=tf.int32, name='input_ids')
        input_mask = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32, name='input_mask')
        segment_ids = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32, name='segment_ids')
        label_ids = tf.placeholder(
            shape=[None, args.max_seq_length], dtype=tf.int32, name='label_ids')
        is_training = tf.get_variable(
            "is_training", shape=[], dtype=tf.bool, trainable=False)

        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            len(label_list) + 1, False, args.dropout_rate, args.lstm_size, args.cell, args.num_layers)

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


def predict(args, processor, tokenizer, bert_config, sess_config, label_list):
    """
    预测函数
    """
    # 生成3个examples
    predict_examples = processor.get_test_examples(args.data_dir)
    predict_file = os.path.join(args.output_dir, "predict.tf_record")
    filed_based_convert_examples_to_features(
        predict_examples, label_list, args.max_seq_length,
        tokenizer, predict_file, args.output_dir, mode="test")
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
        saver.restore(sess, tf.train.latest_checkpoint(save_dir))
        # 打印张量名
        # tensor_list = [
        #     n.name for n in tf.get_default_graph().as_graph_def().node if 'ReverseSequence' in n.name]
        # print(tensor_list)
        # 通过张量名获取模型的占位符和参数
        input_ids = tf.get_default_graph().get_tensor_by_name('input_ids:0')
        input_mask = tf.get_default_graph().get_tensor_by_name('input_mask:0')
        segment_ids = tf.get_default_graph().get_tensor_by_name('segment_ids:0')
        label_ids = tf.get_default_graph().get_tensor_by_name('label_ids:0')
        sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name(
            'is_training:0'), tf.constant(False, dtype=tf.bool)))
        # 找到crf输出, 注意其名称在crf_decode源码中, 可以在graph中查到
        pred_ids = tf.get_default_graph().get_tensor_by_name('ReverseSequence_1:0')



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
        "ner": NerProcessor
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
        predict(args=args, processor=processor, tokenizer=tokenizer,
                bert_config=bert_config, sess_config=session_config, label_list=label_list)
