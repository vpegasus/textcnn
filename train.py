#!/usr/bin/env python  
# -*- coding: utf-8 -*-
"""
@author: Prince
@file: train.py 
@time: 2018-2-26 07: 51
@license: Apache License
@contact: pegasus.wenjia@foxmail.com 
"""

import tensorflow as tf
from config import flag
from textcnn import TextCNN


def train(inputs, labels):
    """

    :param input:
    :return:
    """
    with tf.Session() as sess:
        cnn = TextCNN(flag.setence, flag.num_classes, flag.vocab_size, flag.embedding_size, flag.filter_sizes,
                      flag.num_filters, flag.keep_prob)

        output = cnn(inputs)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=labels)
        total_loss = loss + flag.decay_rate * tf.nn.l2_loss(cnn.final_weight + cnn.final_bias)
        global_step = tf.train.get_or_create_global_step()

        optimizer = tf.train.AdamOptimizer(flag.learning_rate)

        gradients_vars = optimizer.compute_gradients(total_loss)

        for i, (grad, var) in enumerate(gradients_vars):
            if grad is not None:
                gradients_vars[i] = (tf.clip_by_value(grad, -10, 10), var)
                tf.summary.histogram(var.name + '/grad', grad)  # tf.histogram_summary
        tf.summary.scalar('loss', total_loss)
        sum_merge = tf.summary.merge_all()
        train_op = optimizer.apply_gradients(gradients_vars, global_step=global_step)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(flag.model_saved_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('reloading model parameters..')
        else:
            print('create mdoel from scratch..')
            sess.run(tf.global_variables_initializer())

        summarizer = tf.summary.FileWriter(flag.model_saved_dir, sess.graph)

        for i in range(flag.num_loop):
            step_loss,summary, _ = sess.run([total_loss,sum_merge, train_op])
            if i %1000 == 0:
                print('check points {}'.format(i))
                saver.save(sess,flag.model_saved_path, global_step=global_step)
                summarizer.add_summary(summary,global_step=global_step)
                
