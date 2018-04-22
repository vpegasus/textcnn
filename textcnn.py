#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: textcnn.py
@time: 2018-2-26 00: 17
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""

import tensorflow as tf


class TextCNN(object):
    def __init__(self, setence_lenghth, num_classes, vocab_size, embedding_size, filter_sizes, num_filters,
                 keep_prob_of_dropout=1.0):
        """

        :param setence_lenghth:
        :param num_classes:
        :param vocab_size:
        :param embedding_size:
        :param filter_sizes:
        :param num_filters:
        :param keep_prob_of_dropout:
        """

        self.setence_length = setence_lenghth
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filter_sizes = filter_sizes
        self.filter_list_length = len(self.filter_sizes)
        self.num_filters = num_filters
        self.total_filters = self.num_filters * self.filter_list_length
        self.keep_prob_of_dropout = keep_prob_of_dropout

        self.final_weight = tf.get_variable(name='final_weight', shape=[self.total_filters, num_classes],
                                            initializer=tf.contrib.layers.xavier_initializer())
        self.final_bias = tf.get_variable(name='final_bias', shape=[num_classes],
                                          initializer=tf.constant_initializer(0.1))
        self.conv_outputs = []

        self.create_embedding_matrix()

    def create_embedding_matrix(self):
        self.embedding_matrix = tf.get_variable(name='embedding_matrix', shape=[self.vocab_size, self.embedding_size],
                                                initializer=tf.random_uniform_initializer(-1.0, 1.0))

    def embedding(self, input):
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, input)
        self.embedding_output = tf.expand_dims(embedded, -1)
        return self.embedding_output

    def conv_layer(self):

    def _conv_loop_body(self, filter_idx):
        with tf.name_scope('conv_op_{}'.format(filter_idx)):
            filter_size = self.filter_sizes[filter_idx]
            filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
        conv_filter = tf.get_variable('conv_filter', shape=filter_shape,
                                 initializer=tf.truncated_normal_initializer(0.0, stddev=0.1))
        filter_bias = tf.get_variable('filter_bias', shape=[self.num_filters], initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(self.embedding_output, conv_filter, strides=[1, 1, 1, 1], padding='VALID', name='conv')
        act = tf.nn.relu(tf.nn.bias_add(conv, filter_bias), name='relu')
        max_pool = tf.nn.max_pool(act, ksize=[1, self.setence_length - self.filter_size - 1, 1, 1],
                                  strides=[1, 1, 1, 1])
        self.conv_outputs.append(max_pool)
        return filter_idx + 1

    def loop(self):
        filter_idx = 0
        tf.while_loop(lambda filter_idx: tf.less(filter_idx, self.filter_list_length),
                      self._conv_loop_body,
                      loop_vars=[filter_idx])

    def get_conv_output(self):
        return self.conv_outputs

    def concat_and_flat(self):
        res = tf.concat(self.conv_outputs, 3)
        return tf.reshape(res, [-1, self.total_filters])

    def drop_out(self, input):
        return tf.nn.dropout(input, keep_prob=self.keep_prob_of_dropout)

    def final_fc(self, input):
        res = tf.nn.xw_plus_b(input, self.final_weight, self.final_bias)
        # res  = tf.argmax(res,1,name= 'predictions')
        return res

    def __call__(self, input):
        _ = self.embedding(input)
        self.loop()
        res = self.concat_and_flat()
        res2 = self.drop_out(res)
        res3 = self.final_fc(res2)
        return res3
