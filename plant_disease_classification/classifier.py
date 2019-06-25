#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

image_size = 128
num_channels = 3

graph_path = os.path.join('plant_disease_classification/ckpts/', 'plants-disease-model.meta')
checkpoint_path = os.path.join('plant_disease_classification/ckpts/', 'plants-disease-model')

session = tf.compat.v1.Session()
saver = tf.compat.v1.train.import_meta_graph(graph_path)
saver.restore(session, tf.train.latest_checkpoint('plant_disease_classification/ckpts/'))
# saver.restore(session, checkpoint_path)

def classify(file_path='plant_disease_classification/datasets/test/0a02f9b47e8082558fa257092f0cedee.jpg'):
    print(file_path)
    images = []
    image = cv2.imread(file_path)
    # Resizing the image to our desired size and
    # preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0 / 255.0)
    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size, image_size, num_channels)
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")
    # Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((1, 38))
    feed_dict_testing = {x: x_batch, y_true: y_test_images}
    result = session.run(y_pred, feed_dict=feed_dict_testing)
    print(result)
