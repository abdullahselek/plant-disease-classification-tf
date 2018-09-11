#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf

image_size = 128
num_channels = 3

session = tf.Session()
saver = tf.train.import_meta_graph('plants-disease-model.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))

def predict(filename):
    images = []
    image = cv2.imread(filename)
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
    y_test_images = np.zeros((1, 2))
