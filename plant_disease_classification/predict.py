#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf

image_size = 128

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
