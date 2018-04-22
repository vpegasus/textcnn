#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: config.py 
@time: 2018-2-26 07: 55
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""
import tensorflow as tf

flag = tf.flags.FLAGS

tf.flags.DEFINE_float('learning_rate', 0.001, 'the learning rate of optimizer')
tf.flags.DEFINE_float('decay_rate', 0.01, 'the decay rate of l2 loss ')
tf.flags.DEFINE_integer('sentence_length', 500, 'the length of a doc')
tf.flags.DEFINE_integer('num_classes', 18, 'the number of classes')
tf.flags.DEFINE_integer('vocab_size', 5000, 'the capacity of dict')
tf.flags.DEFINE_integer('filter_sizes', 200, 'the list of filter sizes')
tf.flags.DEFINE_integer('num_filters', 8, 'how many filters of a convolution to use')
tf.flags.DEFINE_float('keep_prob', 0.7, 'the ratio weights to keep of one training step ')

tf.flags.DEFINE_integer('loop_num', 100000, 'the number of loops')
