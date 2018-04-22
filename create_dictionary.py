#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: create_dictionary.py 
@time: 2018/2/26 14:09
@license: Apache License
@contact: mawenjia@021.com 
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: pre_process.py
@time: 202018/2/5 10: 46
@license: Apache License
@contact: mawenjia@021.com
"""

import pandas as pd

import jieba.posseg as pseg
import pickle
from gensim import corpora
import time
import jieba

import re
import numpy as np


def build_dictionary(assistant_dict_dir, data_dir, new_dict_dir):
    """

    :param assistant_dict_dir:
    :param data_dir:
    :param new_dict_dir:
    :return:
    """
    jieba.load_userdict(assistant_dict_dir)
    df = pd.read_csv(data_dir, sep='\t', names=['url', 'title', 'content', 'genre', 'source'])  # todo nrows
    start = time.time()
    ns, vs = [], []
    doc_lens = []
    # key_id = []
    # genres = []
    temp = 0
    loop = 0
    while True:
        loop += 1
        if loop % 10 == 0:
            temp = time.time()
            print('loop: {}/ {}, duration: {:.3f} minutes.'.format(loop,df.shape[0], (temp - start) / 60),end='\r')
        try:
            # data = df.get_chunk(1)
            # key_id.append(extract_id(data.url.values[0]))
            # genres.append(data.genre.values[0])
            # seg = segmentation(data.content.values[0])
            data = df.iloc[loop-1]
            if data.content is np.nan:
                continue
            seg = segmentation(data.content)
            n, v, doc_len = extract_n_v(seg)
            doc_lens.append(doc_len)
            ns.append(n)
            vs.append(v)
        except:
            print('loop: {}/ {}, duration: {:.3f} minutes.'.format(loop, df.shape[0], (temp - start) / 60))
            print('extract nouns and verbs done.')
            break
    dictionary = build_n_v_dict(ns, vs)
    print('loop {}, consume {:.3f} minutes.'.format(loop, (temp - start) / 60))
    # ni, vi = convert2ids(ns, vs, dictionary)
    with open(new_dict_dir, 'wb') as output:
        pickle.dump(dictionary, output, -1)
        print('new_dict saved.')
    return dictionary  # , ni, vi, key_id, genres


def extract_id(url):
    pattern = '.*?mobile/(.*?).html'
    res = re.findall(pattern, url)[0]
    return res


def segmentation(data):
    seg = pseg.cut(data)
    return seg


def extract_n_v(pseg_cut_data):
    noun = []
    verb = []
    doc_len = 0
    for w, p in pseg_cut_data:
        if p[0] == 'n':
            noun.append(w)
        if p[0] == 'v':
            verb.append(w)
        doc_len += 1
    return noun, verb, doc_len


def build_n_v_dict(nouns, verbs):
    dictionary = corpora.Dictionary(nouns + verbs)
    dictionary.filter_extremes(no_above=0.8)
    return dictionary


def convert2ids(nouns, verbs, lexicon):
    """

    :param nouns: [[txt],[txt2]]
    :param verbs:
    :param lexicon:
    :return:
    """
    noun_ids = [lexicon.doc2bow(txt) for txt in nouns]
    verb_ids = [lexicon.doc2bow(txt) for txt in verbs]
    return noun_ids, verb_ids


def extract(news, dictionary):
    """
    a module to extract info from a raw news
    :param news: string
    :param dictionary:
    :return:
    """
    seg = segmentation(news)
    nouns, verbs, news_len = extract_n_v(seg)
    noun_ids, verb_ids = convert2ids([nouns], [verbs], dictionary)
    return noun_ids, verb_ids, news_len


if __name__ == '__main__':

    local_dict_dir = 'D:\\Projects\\hotwing2.0\\model_helper\\dictionary_20180223.txt'
    data_dir = 'e:/data/for_textcnn/total_data.csv'
    new_dict_dir = 'e:/data/for_textcnn/dictionary.pkl'
    dictionary = build_dictionary(local_dict_dir,
                                  data_dir,
                                  new_dict_dir)
