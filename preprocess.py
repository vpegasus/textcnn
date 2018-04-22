#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: preprocess.py 
@time: 2018/2/26 14:15
@license: Apache License
@contact: mawenjia@021.com 
"""

import os
import pandas as pd





datalist1 = os.listdir('e:/data/for_textcnn/train')
subdirs=['train','test']

total_data = pd.DataFrame()

for subdir in subdirs:
    for file in datalist1:
        print('processing {} / {}..'.format(subdir,file))
        file_dir = os.path.join('e:/data/for_textcnn/',subdir,file)
        df = pd.read_csv(file_dir,sep='\t',names = ['url','title','content','class','source'])
        total_data = pd.concat([total_data,df])

total_data.to_csv('e:/data/for_textcnn/total_data.csv',sep='\t',header=False,index=False)
print('concatenate finished.')

