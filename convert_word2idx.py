#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: convert_word2idx.py 
@time: 2018/2/26 17:17
@license: Apache License
@contact: mawenjia@021.com 
"""
import pickle
import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv('/home/prince/PycharmProjects/sentiment_analysis_textcnn-master/data/cutclean_label_corpus10000.txt',
                 sep='\t', names=['genre', 'content'])
t = ''
for i in range(df.shape[0]):
    t += df.iloc[i].content

words = Counter(t)

sorted_words = words.most_common(len(words))
dictionary = dict()

for char, freq in sorted_words:
    dictionary[char] = len(dictionary)

df['wd2idx'] = df.apply(lambda x: [dictionary[i] for i in x.content], axis = 1)

data = list(df.wd2idx)
labels = list(df.genre)
