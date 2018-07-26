#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from plant_disease_detection import datagenerator

train_path = os.path.join('plant_disease_detection', 'datasets/train')
classes = os.listdir(train_path)
num_classes = len(classes)

# 20% of the data will automatically be used for validation
validation_size = 0.2
img_size = 128
num_channels = 3

session = tf.Session()
x = tf.placeholder(tf.float32,
                   shape=[None, img_size, img_size, num_channels],
                   name='x')

# labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def get_data():
    # Load and read training data
    data = datagenerator.read_train_sets(train_path, img_size, classes, validation_size)
    print('Complete reading input data. Will Now print a snippet of it')
    print('Number of files in Training-set:\t\t{}'.format(len(data.train.labels)))
    print('Number of files in Validation-set:\t{}'.format(len(data.valid.labels)))
    return data

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
