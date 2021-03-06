# -*- coding: utf-8 -*-
"""
Postpreccessing
@author:yuhaitao
"""
import sys
import json
import os
import zipfile
import traceback
import argparse
from tqdm import tqdm


SUCCESS = 0
FILE_ERROR = 1
ENCODING_ERROR = 2
JSON_ERROR = 3
SCHEMA_ERROR = 4
TEXT_ERROR = 5
CODE_INFO = ['success', 'file_reading_error', 'encoding_error', 'json_parse_error',
             'schema_error', 'input_text_not_in_dataset']


def del_bookname(entity_name):
    """delete the book name"""
    if entity_name.startswith(u'《') and entity_name.endswith(u'》'):
        entity_name = entity_name[1:-1]
    return entity_name


def load_predict_result(predict_filename):
    """Loads the file to be predicted"""
    predict_result = {}
    ret_code = SUCCESS
    # try:
    #     predict_file_zip = zipfile.ZipFile(predict_filename)
    # except:
    #     ret_code = FILE_ERROR
    #     return predict_result, ret_code
    # for predict_file in predict_file_zip.namelist():
    with open(predict_filename) as f:
        for line in f:
            try:
                line = line.strip()
            except:
                ret_code = ENCODING_ERROR
                return predict_result, ret_code
            try:
                json_info = json.loads(line)
            except:
                ret_code = JSON_ERROR
                return predict_result, ret_code
            if 'text' not in json_info or 'spo_list' not in json_info:
                ret_code = SCHEMA_ERROR
                return predict_result, ret_code
            sent = json_info['text']
            spo_set = set()
            for spo_item in json_info['spo_list']:
                if type(spo_item) is not dict or 'subject' not in spo_item \
                        or 'predicate' not in spo_item \
                        or 'object' not in spo_item or \
                        not isinstance(spo_item['subject'], str) or \
                        not isinstance(spo_item['object'], str):
                    ret_code = SCHEMA_ERROR
                    return predict_result, ret_code
                s = del_bookname(spo_item['subject'].lower())
                o = del_bookname(spo_item['object'].lower())
                spo_set.add((s, spo_item['predicate'], o))
            predict_result[sent] = spo_set
    return predict_result, ret_code


def load_test_dataset(golden_filename):
    """load golden file"""
    golden_dict = {}
    ret_code = SUCCESS
    with open(golden_filename) as gf:
        for line in gf:
            try:
                line = line.strip()
            except:
                ret_code = ENCODING_ERROR
                return golden_dict, ret_code
            try:
                json_info = json.loads(line)
            except:
                ret_code = JSON_ERROR
                return golden_dict, ret_code
            try:
                sent = json_info['text']
                spo_list = json_info['spo_list']
            except:
                ret_code = SCHEMA_ERROR
                return golden_dict, ret_code

            spo_result = []
            for item in spo_list:
                o = del_bookname(item['object'].lower())
                s = del_bookname(item['subject'].lower())
                spo_result.append((s, item['predicate'], o))
            spo_result = set(spo_result)
            golden_dict[sent] = spo_result
    return golden_dict, ret_code


def load_dict(dict_filename):
    """load alias dict"""
    alias_dict = {}
    ret_code = SUCCESS
    if dict_filename == "":
        return alias_dict, ret_code
    try:
        with open(dict_filename) as af:
            for line in af:
                line = line.decode().strip()
                words = line.split('\t')
                alias_dict[words[0].lower()] = set()
                for alias_word in words[1:]:
                    alias_dict[words[0].lower()].add(alias_word.lower())
    except:
        ret_code = FILE_ERROR
    return alias_dict, ret_code


def is_spo_correct(spo, golden_spo_set, alias_dict, loc_dict):
    """if the spo is correct"""
    if spo in golden_spo_set:
        return True
    (s, p, o) = spo
    # alias dictionary
    s_alias_set = alias_dict.get(s, set())
    s_alias_set.add(s)
    o_alias_set = alias_dict.get(o, set())
    o_alias_set.add(o)
    for s_a in s_alias_set:
        for o_a in o_alias_set:
            if (s_a, p, o_a) in golden_spo_set:
                return True
    for golden_spo in golden_spo_set:
        (golden_s, golden_p, golden_o) = golden_spo
        golden_o_set = loc_dict.get(golden_o, set())
        for g_o in golden_o_set:
            if s == golden_s and p == golden_p and o == g_o:
                return True
    return False


def calc_pr(predict_filename, alias_filename, location_filename,
            golden_filename):
    """calculate precision, recall, f1"""
    ret_info = {}
    # # load location dict
    # loc_dict, ret_code = load_dict(location_filename)
    # if ret_code != SUCCESS:
    #     ret_info['errorCode'] = ret_code
    #     ret_info['errorMsg'] = CODE_INFO[ret_code]
    #     print >> sys.stderr, 'loc file is error'
    #     return ret_info

    # # load alias dict
    # alias_dict, ret_code = load_dict(alias_filename)
    # if ret_code != SUCCESS:
    #     ret_info['errorCode'] = ret_code
    #     ret_info['errorMsg'] = CODE_INFO[ret_code]
    #     print >> sys.stderr, 'alias file is error'
    #     return ret_info
    # load test dataset
    golden_dict, ret_code = load_test_dataset(golden_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print(sys.stderr, 'golden file is error')
        return ret_info
    # load predict result
    predict_result, ret_code = load_predict_result(predict_filename)
    if ret_code != SUCCESS:
        ret_info['errorCode'] = ret_code
        ret_info['errorMsg'] = CODE_INFO[ret_code]
        print(sys.stderr, 'predict file is error')
        return ret_info

    # evaluation
    alias_dict = {}
    loc_dict = {}
    correct_sum, predict_sum, recall_sum = 0.0, 0.0, 0.0
    for sent in golden_dict:
        golden_spo_set = golden_dict[sent]
        predict_spo_set = predict_result.get(sent, set())

        recall_sum += len(golden_spo_set)
        predict_sum += len(predict_spo_set)
        for spo in predict_spo_set:
            if is_spo_correct(spo, golden_spo_set, alias_dict, loc_dict):
                correct_sum += 1
    print(sys.stderr, 'correct spo num = ', correct_sum)
    print(sys.stderr, 'submitted spo num = ', predict_sum)
    print(sys.stderr, 'golden set spo num = ', recall_sum)
    precision = correct_sum / predict_sum if predict_sum > 0 else 0.0
    recall = correct_sum / recall_sum if recall_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall > 0 else 0.0
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)
    ret_info['errorCode'] = SUCCESS
    ret_info['errorMsg'] = CODE_INFO[SUCCESS]
    ret_info['data'] = []
    ret_info['data'].append({'name': 'precision', 'value': precision})
    ret_info['data'].append({'name': 'recall', 'value': recall})
    ret_info['data'].append({'name': 'f1-score', 'value': f1})
    return ret_info


def get_type_dic(trian_file, has_type_dic=True):
    type_dic = {}
    if not has_type_dic:
        with open(trian_file, 'r') as f:
            for line in f:
                line = line.strip()
                dic = json.loads(line)
                for spo in dic['spo_list']:
                    if spo['predicate'] not in type_dic:
                        type_dic[spo['predicate']] = {
                            'subject_type': spo['subject_type'], 'object_type': spo['object_type']}
                if len(type_dic) == 50:
                    break

        with open(args.type_dic_file, 'w') as f:
            for k, v in type_dic.items():
                f.write(k + '\t' + v['subject_type'] +
                        '\t' + v['object_type'] + '\n')
    else:
        with open(args.type_dic_file, 'r') as f:
            for line in f:
                type_list = line.strip().split()
                if len(type_list) != 3:
                    raise ValueError
                type_dic[type_list[0]] = {
                    'subject_type': type_list[1], 'object_type': type_list[2]}
    return type_dic


def generate_result_file(golden_file, predict_file, eng_label_dic, type_dic, result_dir, mode='test'):
    """
    生成result文件
    """
    f_pre = open(predict_file, 'r')
    f_res = open(os.path.join(result_dir, (mode + '_result.json')),
                 'w', encoding='utf-8')
    with open(golden_file, 'r') as f:
        pre_list = f_pre.readline().strip().split()
        for line in tqdm(f):
            dic = json.loads(line.strip())
            dic_res = {"text": dic['text'], "spo_list": []}
            sentence_ori = ''.join(s.strip() for s in dic['text'].split())
            while pre_list and pre_list[0] == sentence_ori[0:len(pre_list[0])]:
                if pre_list[4] != 'NORELATION' and pre_list[4][0:2] != 'RE':
                    dic_res["spo_list"].append({
                        "predicate": eng_label_dic[pre_list[4]],
                        "subject": pre_list[1],
                        "subject_type": type_dic[eng_label_dic[pre_list[4]]]['subject_type'],
                        "object": pre_list[2],
                        "object_type": type_dic[eng_label_dic[pre_list[4]]]['object_type']
                    })
                pre_list = f_pre.readline().strip().split()
            res = json.dumps(dic_res, ensure_ascii=False)
            f_res.write(res + '\n')
    f_pre.close()
    f_res.close()


def postprocess(golden_dir, predict_dir, eng_label_dic_file, result_dir, has_type_dic=True):
    # 先获取type_dic
    type_dic = get_type_dic(os.path.join(
        golden_dir, 'train_data.json'), has_type_dic)
    # print(type_dic)
    eng_label_dic = {}
    with open(eng_label_dic_file, 'r') as f:
        for line in f:
            item = line.strip().split()
            if len(item) != 2:
                raise ValueError
            eng_label_dic[item[1]] = item[0]
    # 分别生成dev集和test集的result文件
    generate_result_file(os.path.join(golden_dir, 'dev_data.json'), os.path.join(
        predict_dir, 'prediction_dev.txt'), eng_label_dic, type_dic, result_dir, mode='dev')
    generate_result_file(os.path.join(golden_dir, 'test1_data_postag.json'), os.path.join(
        predict_dir, 'prediction_test.txt'), eng_label_dic, type_dic, result_dir, mode='test')


if __name__ == '__main__':
    # reload(sys)
    # sys.setdefaultencoding('utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default='demo/')
    parser.add_argument("--golden_dir", type=str, default='../data/ori_data/')
    parser.add_argument("--predict_dir", type=str, default='../data/')
    parser.add_argument("--result_dir", type=str, default='../data/RC_result/')
    parser.add_argument("--eng_label_dic_file",
                        type=str, default='../dict/p_eng')
    parser.add_argument("--type_dic_file", type=str,
                        default='../dict/type_dic')
    parser.add_argument("--golden_file", type=str,
                        default='', help="true spo results")
    parser.add_argument("--predict_file", type=str,
                        default='', help="spo results predicted")
    parser.add_argument("--loc_file", type=str,
                        default='', help="location entities of various granularity")
    parser.add_argument("--alias_file", type=str,
                        default='', help="entities alias dictionary")
    args = parser.parse_args()
    args.predict_dir = os.path.join(
        args.predict_dir, 'RC_model_' + args.experiment_name)
    # 生成dev和test的结果文件
    postprocess(args.golden_dir, args.predict_dir,
                args.eng_label_dic_file, args.result_dir, has_type_dic=True)

    # # 计算F1
    ret_info = calc_pr(os.path.join(args.result_dir, 'dev_result.json'),
                       '', '', os.path.join(args.golden_dir, 'dev_data.json'))
    print(json.dumps(ret_info))
