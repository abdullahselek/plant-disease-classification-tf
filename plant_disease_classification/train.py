#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from plant_disease_classification import datagenerator

train_path = os.path.join('plant_disease_classification', 'datasets/train')
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

def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # Define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size,
                                    conv_filter_size,
                                    num_input_channels,
                                    num_filters])
    # Create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    # Creating the convolutional layer.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')
    layer += biases
    # Use max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)
    return layer

def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    # Connected layer takes input x and produces wx+b.Since,
    # these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer
