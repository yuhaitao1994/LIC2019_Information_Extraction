# coding=utf-8
"""
数据读取预处理，定义一个data reader类，生成CSV文件，供bert的data processor读取
"""
import json
import os
import codecs
import sys
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

        # self._feature_dict = {}
        # self._feature_dict['postag_dict'] = \
        #     self._load_dict_from_file(self._dict_path_dict['postag_dict'])
        # self._feature_dict['label_dict'] = \
        #     self._load_label_dict(self._dict_path_dict['label_dict'])
        # # 将之前所有的字典反向
        # self._reverse_dict = {name: self._get_reverse_dict(name) for name in
        #                       self._dict_path_dict.keys()}
        # self._reverse_dict['eng_map_p_dict'] = self._reverse_p_eng(
        #     self._p_map_eng_dict)
        # self._UNK_IDX = 0

        # 统计在所有训练数据中主体和客体所覆盖的postag
        # 主体subject，客体object
        # self.subject_tags, self.object_tags = self.count_tags(
        #     self.train_data_list_path, self._postag_dict_path)

    def _load_label_dict(self, dict_name):
        """load label dict from file"""
        label_dict = {}
        with codecs.open(dict_name, 'r', 'utf-8') as fr:
            for idx, line in enumerate(fr):
                p, p_eng = line.strip().split('\t')
                label_dict[p_eng] = idx
                self._p_map_eng_dict[p] = p_eng
        return label_dict

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

    def _get_feed_iterator(self, line, need_input=False, need_label=True):
        """
        生成一条数据，应修改为对每一个line生成一个样本列表，其中除了正样本，每个line还生成一个负样本
        """
        # verify that the input format of each line meets the format
        if not self._is_valid_input_data(line):
            print >> sys.stderr, 'Format is error'
            return None
        dic = json.loads(line)
        sentence = dic['text']

        pos_list = ['nz', 'ns', 'nw', 'nr', 'nt']

        token_list = []
        label_list = []
        sentence = ''.join(s.strip() for s in sentence.split())
        for s in sentence:
            token_list.append(s)
        if not need_label:
            label_list = ['X'] * len(token_list)
            return token_list, label_list
        else:
            label_list = ['O'] * len(token_list)
            entity_set = set()
            for spo in dic['spo_list']:
                entity_set.add(spo['subject'])
                entity_set.add(spo['object'])
            for entity in entity_set:
                try:
                    index = sentence.index(entity)
                    flag = 0
                    for i in range(index, index + len(entity)):
                        if label_list[i] == 'O':
                            flag = 1
                            break
                    if flag == 0:
                        continue
                    else:
                        for i in range(len(entity)):
                            if i == 0:
                                label_list[index + i] = 'B'
                            elif i == len(entity) - 1:
                                label_list[index + i] = 'E'
                            else:
                                label_list[index + i] = 'I'
                except:
                    continue
            # 根据postag，给更多实体标注
            another_set = set()
            for postag in dic["postag"]:
                e = postag["word"]
                if postag["pos"] in pos_list:
                    another_set.add(e)
            another_set = another_set - entity_set

            for e in another_set:
                try:
                    index = sentence.index(e)
                    flag = 0
                    for i in range(index, index + len(e)):
                        if label_list[i] != 'O':
                            flag = 1
                            break
                    if flag == 1:
                        continue
                    else:
                        for i in range(len(e)):
                            if label_list[index + i] == 'O':
                                if i == 0:
                                    label_list[index + i] = 'B'
                                elif i == len(e) - 1:
                                    label_list[index + i] = 'E'
                                else:
                                    label_list[index + i] = 'I'
                except:
                    continue

            return token_list, label_list

    def path_reader(self, data_path, need_input=False, need_label=True):
        """Read data from data_path"""
        def reader():
            """Generator"""
            if os.path.isdir(data_path):
                input_files = os.listdir(data_path)
                for data_file in input_files:
                    data_file_path = os.path.join(data_path, data_file)
                    for line in open(data_file_path.strip()):
                        token_list, label_list = self._get_feed_iterator(
                            line.strip(), need_input, need_label)
                        if token_list is None:
                            continue
                        yield token_list, label_list
            elif os.path.isfile(data_path):
                for line in open(data_path.strip()):
                    # 对文件每一行生成数据
                    token_list, label_list = self._get_feed_iterator(
                        line.strip(), need_input, need_label)
                    if token_list is None:
                        continue
                    yield token_list, label_list

        return reader

    def count_tags(self, train_file, postag_file):
        """
        统计所有主客体覆盖到的postag类别
        """
        subject_tags, object_tags = {}, {}

        # 如果文件存在
        if os.path.isfile('../dict/sub_tag') and os.path.isfile('../dict/obj_tag'):
            with open('../dict/sub_tag', 'r') as fs:
                for line in fs:
                    key, value = line.strip().split('\t')
                    subject_tags[key] = int(value)
            with open('../dict/obj_tag', 'r') as fo:
                for line in fo:
                    key, value = line.strip().split('\t')
                    object_tags[key] = int(value)
            return subject_tags, object_tags

        # 如果文件不存在
        with open(postag_file, 'r') as f:
            for line in f:
                tag = line.strip()
                subject_tags[tag], object_tags[tag] = 0, 0
        print("开始统计主客体的postag类别...")
        with open(train_file, 'r') as f:
            for line in tqdm(f):
                dic = json.loads(line.strip())
                for spo in dic['spo_list']:
                    for postag in dic['postag']:
                        if postag['word'] == spo['subject']:
                            subject_tags[postag['pos']] += 1
                            break
                    for postag in dic['postag']:
                        if postag['word'] == spo['object']:
                            object_tags[postag['pos']] += 1
                            break
        s = list(subject_tags.keys())
        o = list(object_tags.keys())
        for key in s:
            if subject_tags[key] == 0:
                del subject_tags[key]
        for key in o:
            if object_tags[key] == 0:
                del object_tags[key]
        with open('../dict/sub_tag', 'w') as fs:
            for key, value in subject_tags.items():
                fs.write(key + '\t' + str(value) + '\n')
        with open('../dict/obj_tag', 'w') as fo:
            for key, value in object_tags.items():
                fo.write(key + '\t' + str(value) + '\n')

        return subject_tags, object_tags

    def get_train_reader(self, need_input=False, need_label=True):
        """Data reader during training"""
        return self.path_reader(self.train_data_list_path, need_input, need_label)

    def get_dev_reader(self, need_input=True, need_label=True):
        """Data reader during dev"""
        return self.path_reader(self.dev_data_list_path, need_input, need_label)

    def get_test_reader(self, test_file_path='', need_input=True, need_label=False):
        """Data reader during predict"""
        return self.path_reader(test_file_path, need_input, need_label)

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
        dev_data_list_path='../data/ori_data/dev_data.json')

    # prepare data reader
    train = data_generator.get_train_reader()
    with open("../data/NER_data/train.txt", 'w') as f:
        for token_list, label_list in tqdm(train()):
            for i in range(len(token_list)):
                if token_list[i] == '' and label_list[i] == '':
                    raise ValueError
                f.write(str(token_list[i]) + ' ' + str(label_list[i]) + '\n')
            f.write('\n')

    dev = data_generator.get_dev_reader()
    index = 0
    with open("../data/NER_data/dev.txt", 'w') as f:
        for token_list, label_list in tqdm(dev()):
            index += 1
            for i in range(len(token_list)):
                f.write(str(token_list[i]) + ' ' + str(label_list[i]) + '\n')
            f.write('\n')
    print('index:{}'.format(index))

    test = data_generator.get_test_reader(
        test_file_path='../data/ori_data/test1_data_postag.json')
    with open("../data/NER_data/test.txt", 'w') as f:
        for token_list, label_list in tqdm(test()):
            for i in range(len(token_list)):
                f.write(str(token_list[i]) + ' ' + str(label_list[i]) + '\n')
            f.write('\n')
