# coding=utf-8
"""
数据读取预处理，SO label
"""
import json
import os
import codecs
import sys
import re
import random
import pandas as pd
import numpy as np
from tqdm import tqdm


class MyDataReader(object):
    """
    class for my data reader
    """

    def __init__(self,
                 postag_dict_path,
                 label_dict_path,
                 train_data_list_path='',
                 dev_data_list_path='',
                 train_pc_file='',
                 dev_pc_file=''):
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self.train_data_list_path = train_data_list_path
        self.dev_data_list_path = dev_data_list_path
        self.train_pc_file = train_pc_file
        self.dev_pc_file = dev_pc_file
        self._p_map_eng_dict = {}

        # 统计各种类别数据数量的词典
        self.label_num_dic = {}
        # load dictionary
        self._dict_path_dict = {'postag_dict': self._postag_dict_path,
                                'label_dict': self._label_dict_path}
        # check if the file exists
        for input_dict in [postag_dict_path,
                           label_dict_path, train_data_list_path, dev_data_list_path]:
            if not os.path.exists(input_dict):
                raise ValueError("%s not found." % (input_dict))
                return

        self._feature_dict = {}
        self._feature_dict['postag_dict'] = \
            self._load_dict_from_file(self._dict_path_dict['postag_dict'])
        self._feature_dict['label_dict'], self.label_eng_dict = \
            self._load_label_dict(self._dict_path_dict['label_dict'])
        print(self.label_eng_dict)
        # 将之前所有的字典反向
        self._reverse_dict = {name: self._get_reverse_dict(name) for name in
                              self._dict_path_dict.keys()}
        self._reverse_dict['eng_map_p_dict'] = self._reverse_p_eng(
            self._p_map_eng_dict)
        self._UNK_IDX = 0

        # 统计在所有训练数据中主体和客体所覆盖的postag
        # 主体subject，客体object
        # self.subject_tags, self.object_tags = self.count_tags(
        #     self.train_data_list_path, self._postag_dict_path)

    def _load_label_dict(self, dict_name):
        """这个函数重写了"""
        label_dict = {}
        label_to_eng = {}
        pattern = re.compile(r'\s+')
        with codecs.open(dict_name, 'r', 'utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = re.split(pattern, line.strip())
                label_to_eng[p] = p_eng
                label_dict[p_eng] = idx
                self._p_map_eng_dict[p] = p_eng
                self.label_num_dic[p] = 0
                # if p != '木有关系':
                #     label_to_eng['反_' + p] = 'RE_' + p_eng
                #     label_dict['RE_' + p_eng] = idx + 51
                #     self._p_map_eng_dict['反_' + p] = 'RE_' + p_eng
        return label_dict, label_to_eng

    def _load_dict_from_file(self, dict_name, bias=0):
        """
        Load vocabulary from file.
        """
        dict_result = {}
        with codecs.open(dict_name, 'r', 'utf-8') as f_dict:
            for idx, line in enumerate(f_dict):
                line = line.strip()
                dict_result[line] = idx + bias
        return dict_result

    def _cal_mark_single_slot(self, spo_list, sentence):
        """
        Calculate the value of the label
        """
        mark_list = [0] * len(self._feature_dict['label_dict'])
        for spo in spo_list:
            predicate = spo['predicate']
            p_idx = self._feature_dict['label_dict'][self._p_map_eng_dict[predicate]]
            mark_list[p_idx] = 1
        return mark_list

    def _is_valid_input_data(self, input_data):
        """is the input data valid"""
        try:
            dic = json.loads(input_data)
        except:
            return False
        if "text" not in dic or "postag" not in dic or \
                type(dic["postag"]) is not list:
            return False
        for item in dic['postag']:
            if "word" not in item or "pos" not in item:
                return False
        return True

    def _get_feed_iterator(self, line, label_dict, eng_dict, pc_line=None, mode=None):
        """
        生成RC数据
        """
        # verify that the input format of each line meets the format
        if not self._is_valid_input_data(line):
            print(line)
            print(sys.stderr, 'Format is error')
            raise ValueError
        dic = json.loads(line)
        # 注意sentence的长度被截断过
        sentence_ori = ''.join(s.strip() for s in dic['text'].split())

        # 一个样本： text 主体 客体 类别
        sample_list = []
        if mode == 'train':
            for spo in dic['spo_list']:
                sample = sentence_ori
                sample += ('\t' + label_dict[spo['predicate']])
                sample += ('\t' + spo['subject'])
                sample += ('\t' + spo['object'])
                sample_list.append(sample)
        else:
            item = pc_line.strip().split('\t')
            sentence = item[0]
            if sentence != sentence_ori[0:len(sentence)]:
                print(sentence, sentence_ori)
                raise ValueError
            if len(item) == 1:
                return sample_list
            if mode == 'dev':
                for label_eng in item[1:]:
                    label_cns = eng_dict[label_eng]
                    for spo in dic['spo_list']:

            elif mode == 'test':

        return sample_list

    def path_reader(self, data_path, pc_path, mode):
        """Read data from data_path"""
        self._feature_dict['data_keylist'] = []

        def reader():
            """Generator"""
            if mode == "train":
                f = open(data_path.strip())
                for line in f:
                    sample_list = self._get_feed_iterator(line.strip(), label_dict=self.label_eng_dict,
                                                          eng_dict=self._reverse_dict['eng_map_p_dict'], None, mode=mode)
                    if sample_list is None:
                        continue
                    yield sample_list
            else:
                f_pc = open(pc_path, 'r')
                f = open(data_path.strip())
                for line in f:
                    pc_line = f_pc.readline()
                    # 对文件每一行生成数据
                    sample_list = self._get_feed_iterator(line.strip(), label_dict=self.label_eng_dict,
                                                          eng_dict=self._reverse_dict['eng_map_p_dict'], pc_line.strip(), mode=mode)
                    if sample_list is None:
                        continue
                    yield sample_list

        return reader

    def get_train_reader(self, mode='train'):
        """Data reader during training"""
        return self.path_reader(self.train_data_list_path, self.train_pc_file, mode)

    def get_dev_reader(self, mode='dev'):
        """Data reader during dev"""
        return self.path_reader(self.dev_data_list_path, self.dev_pc_file, mode)

    def get_test_reader(self, test_file_path='', test_pc_file='', mode='test'):
        """Data reader during predict"""
        return self.path_reader(test_file_path, test_pc_file, mode)

    def get_dict(self, dict_name):
        """Return dict"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return self._feature_dict[dict_name]

    def get_all_dict_name(self):
        """Get name of all dict"""
        return self._feature_dict.keys()

    def get_dict_size(self, dict_name):
        """Return dict length"""
        if dict_name not in self._feature_dict:
            raise ValueError("dict name %s not found." % (dict_name))
        return len(self._feature_dict[dict_name])

    def _get_reverse_dict(self, dict_name):
        dict_reverse = {}
        for key, value in self._feature_dict[dict_name].items():
            dict_reverse[value] = key
        return dict_reverse

    def _reverse_p_eng(self, dic):
        dict_reverse = {}
        for key, value in dic.items():
            dict_reverse[value] = key
        return dict_reverse

    def get_label_output(self, label_idx):
        """Output final label, used during predict and test"""
        dict_name = 'label_dict'
        if len(self._reverse_dict[dict_name]) == 0:
            self._get_reverse_dict(dict_name)
        p_eng = self._reverse_dict[dict_name][label_idx]
        return self._reverse_dict['eng_map_p_dict'][p_eng]


if __name__ == '__main__':
    # initialize data generator
    data_generator = MyDataReader(
        postag_dict_path='../dict/postag_dict',
        label_dict_path='../dict/p_eng',
        train_data_list_path='../data/ori_data/train_demo.json',
        dev_data_list_path='../data/ori_data/dev_demo.json',
        train_pc_file='',
        dev_pc_file='../data/ori_data/pc_dev.txt')

    # prepare data reader
    train = data_generator.get_train_reader()
    with open("../data/SO_data/train.txt", 'w') as f:
        for sample_list in tqdm(train()):
            for sample in sample_list:
                f.write(sample + '\n')

    dev = data_generator.get_dev_reader()
    with open("../data/SO_data/dev.txt", 'w') as f:
        for sample_list in tqdm(dev()):
            for sample in sample_list:
                f.write(sample + '\n')

    test = data_generator.get_test_reader(
        test_file_path='../data/ori_data/test_demo.json', test_pc_file='../data/ori_data/pc_test.txt')
    with open("../data/SO_data/test.txt", 'w') as f:
        for sample_list in tqdm(test()):
            for sample in sample_list:
                f.write(sample + '\n')

    for key, value in data_generator.label_num_dic.items():
        print(key, value)
