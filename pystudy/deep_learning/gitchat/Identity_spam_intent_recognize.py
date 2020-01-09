#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File : Identity_spam_intent_recognize.py
@Author : jeffsheng
@Date : 2020/1/8
@Desc : 
"""
import os
from pystudy.sysutils.data_utils import read_dictionary
import tensorflow as tf
from pystudy.sysutils.data_utils import get_segmented_vector


class recognize(object):

    def __init__(self,net):
        self.word2id = read_dictionary('./vocab/word2id.pkl')
        net.n_words = len(self.word2id)
        net.build_model()
        self.net=net

    def recognize(self,x_train,y_train):
        with tf.Session() as sess:
            self.net.sess = sess
            sess.run(tf.global_variables_initializer())
            # saver = tf.train.Saver()
            saver = tf.train.import_meta_graph('./model/classifier_save/normal/model.ckpt.meta')
            saver.restore(sess, self.net.model_output_path)
            sum = 0
            from pystudy.sysutils.data_utils import clean_text
            for i in range(len(x_train)):
                predict_class, pred_prob = self.predict(clean_text(x_train[i]), self.word2id._mapping)
                if predict_class == y_train[i]:
                    sum = sum + 1
            print('正确率：', sum / len(x_train))


    def predict(self,text,voca_dict):
        X = get_segmented_vector(text, self.net.max_sequence_length, voca_dict)
        Y = [0]
        f_dict = {
            self.net.inp: X,
            self.net.labels: Y,
            self.net.dropout_keep_prob: 1.0,
        }
        _y, predictions = self.net.sess.run([self.net.soft_max_y, self.net.predict_result], feed_dict=f_dict)
        predict_class = predictions[0]
        pred_prob = _y[0][predict_class]
        return predict_class, pred_prob
