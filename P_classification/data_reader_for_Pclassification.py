# coding=utf-8
"""
数据读取预处理，定义一个data reader类，生成CSV文件，供bert的data processor读取
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
                 dev_data_list_path=''):
        self._postag_dict_path = postag_dict_path
        self._label_dict_path = label_dict_path
        self.train_data_list_path = train_data_list_path
        self.dev_data_list_path = dev_data_list_path
        self._p_map_eng_dict = {}
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
                if p != '木有关系':
                    label_to_eng['反_' + p] = 'RE_' + p_eng
                    label_dict['RE_' + p_eng] = idx + 51
                    self._p_map_eng_dict['反_' + p] = 'RE_' + p_eng
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

    def _get_feed_iterator(self, line, ner_data, label_dict, mode=None):
        """
        生成RC数据
        """
        # verify that the input format of each line meets the format
        if not self._is_valid_input_data(line):
            print(line)
            print(sys.stderr, 'Format is error')
            raise ValueError
        dic = json.loads(line)
        sentence = ner_data['text']
        # 注意sentence的长度被截断过
        sentence_ori = ''.join(s.strip() for s in dic['text'].split())
        if sentence != sentence_ori[0:len(sentence)]:
            print(sentence, sentence_ori)
            raise ValueError

        # 一个样本： text 主体 客体 类别
        sample_list = []
        if mode == 'train':
            """
            训练集完全使用真实体作为positive，postag中的作为假实体negative
            """
            real_entity = []
            fake_entity = []
            # 生成positive sample
            i = 0
            for spo in dic['spo_list']:
                sub = spo['subject']
                obj = spo['object']
                real_entity.append(sub)
                real_entity.append(obj)
                sample_list.append(
                    sentence + '\t' + sub + '\t' + obj + '\t' + label_dict[spo['predicate']])
                if i % 2 == 0:
                    sample_list.append(
                        sentence + '\t' + obj + '\t' + sub + '\t' + label_dict['反_' + spo['predicate']])
                i += 1
            # 生成Negative sample
            # 统计假实体, 使用postag中的, 全英文假实体不要
            for item in dic['postag']:
                if item['pos'] in ['nz', 'ns', 'nw', 'nr', 'nt']:
                    flag = 0
                    for r_e in real_entity:
                        if (item['word'] in r_e) or (r_e in item['word']):
                            flag = 1
                            break
                    if flag == 0 and item['word'].encode('utf-8').isalpha == False:
                        fake_entity.append(item['word'])
                    else:
                        continue

            # i = 0
            # start = -1
            # end = 0
            # while i < len(ner_data['label']):
            #     if ner_data['label'][i] == 'B':
            #         start = i
            #     elif ner_data['label'][i] == 'E':
            #         end = i
            #         if start != -1:
            #             if ''.join(ner_data['text'][start:end + 1]) not in real_entity:
            #                 fake_entity.append(
            #                     ''.join(ner_data['text'][start:end + 1]))
            #     elif ner_data['label'][i] == 'O':
            #         start = -1
            #     i += 1
            # 控制一半的假实体与真实体组成几个negative sample,假实体之间也可以组成negative sample.
            if len(fake_entity) > 0:
                i = 0
                for f_e in fake_entity:
                    random.shuffle(real_entity)
                    if i % 2 == 0:
                        sample_list.append(
                            sentence + '\t' + f_e + '\t' + real_entity[0] + '\t' + label_dict['木有关系'])
                    else:
                        sample_list.append(
                            sentence + '\t' + fake_entity[random.randint(0, i)] + '\t' + f_e + '\t' + label_dict['木有关系'])
                    i += 1
                    if i > int(len(fake_entity) / 2):
                        break

        else:
            entity = []
            i = 0
            start = -1
            end = 0
            while i < len(ner_data['label']):
                if ner_data['label'][i] == 'B':
                    start = i
                elif ner_data['label'][i] == 'E':
                    end = i
                    if start != -1:
                        if ''.join(ner_data['text'][start:end + 1]) not in entity:
                            entity.append(
                                ''.join(ner_data['text'][start:end + 1]))
                elif ner_data['label'][i] == 'O':
                    start = -1
                i += 1

            # 用postag对entity进行修正
            for i in range(len(entity)):
                for item in dic['postag']:
                    if entity[i] in item['word']:
                        entity[i] = item['word']
                        break

            if mode == 'dev':
                for i in range(len(entity)):
                    for j in range(len(entity)):
                        if i == j:
                            continue
                        else:
                            flag = 0
                            for spo in dic['spo_list']:
                                if spo['subject'] in entity[i] and spo['object'] in entity[j]:
                                    sample_list.append(
                                        sentence + '\t' + entity[i] + '\t' + entity[j] + '\t' + label_dict[spo['predicate']])
                                    flag = 1
                                    break
                                elif spo['subject'] in entity[j] and spo['object'] in entity[i]:
                                    sample_list.append(
                                        sentence + '\t' + entity[i] + '\t' + entity[j] + '\t' + label_dict['反_' + spo['predicate']])
                                    flag = 1
                                    break
                            if flag == 0:
                                sample_list.append(
                                    sentence + '\t' + entity[i] + '\t' + entity[j] + '\t' + label_dict['木有关系'])

            elif mode == 'test':
                for i in range(len(entity)):
                    for j in range(len(entity)):
                        if i == j:
                            continue
                        else:
                            sample_list.append(
                                sentence + '\t' + entity[i] + '\t' + entity[j] + '\t' + label_dict['木有关系'])

        return sample_list

    def path_reader(self, data_path, ner_path, mode):
        """Read data from data_path"""
        self._feature_dict['data_keylist'] = []

        def reader():
            """Generator"""
            f_ner = open(ner_path, 'r')
            f = open(data_path.strip())
            for line in f:
                # 选择ner输出文件的一条数据，即text和标注label
                ner_data = {"text": '', "label": []}
                ner_line = f_ner.readline()
                while ner_line.strip():
                    item = ner_line.strip().split(' ')
                    if len(item) != 3:
                        print(item)
                        raise ValueError
                    ner_data['text'] += item[0]
                    ner_data['label'].append(item[2])
                    ner_line = f_ner.readline()

                # 对文件每一行生成数据
                sample_list = self._get_feed_iterator(
                    line.strip(), ner_data, label_dict=self.label_eng_dict, mode=mode)

                if sample_list is None:
                    continue
                yield sample_list

        return reader

    def get_train_reader(self, mode='train'):
        """Data reader during training"""
        return self.path_reader(self.train_data_list_path, self.train_ner_file, mode)

    def get_dev_reader(self, mode='dev'):
        """Data reader during dev"""
        return self.path_reader(self.dev_data_list_path, self.dev_ner_file, mode)

    def get_test_reader(self, test_file_path='', test_ner_file='', mode='test'):
        """Data reader during predict"""
        return self.path_reader(test_file_path, test_ner_file, mode)

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
        train_data_list_path='../data/ori_data/train_data.json',
        dev_data_list_path='../data/ori_data/dev_data.json',
        train_ner_file='../data/ori_data/label_train.txt',
        dev_ner_file='../data/ori_data/label_dev.txt')

    # prepare data reader
    train = data_generator.get_train_reader()
    with open("../data/RC_data/train.txt", 'w') as f:
        for sample_list in tqdm(train()):
            for sample in sample_list:
                f.write(sample + '\n')

    dev = data_generator.get_dev_reader()
    with open("../data/RC_data/dev.txt", 'w') as f:
        for sample_list in tqdm(dev()):
            for sample in sample_list:
                f.write(sample + '\n')

    test = data_generator.get_test_reader(
        test_file_path='../data/ori_data/test1_data_postag.json', test_ner_file='../data/ori_data/label_test.txt')
    with open("../data/RC_data/test.txt", 'w') as f:
        for sample_list in tqdm(test()):
            for sample in sample_list:
                f.write(sample + '\n')
