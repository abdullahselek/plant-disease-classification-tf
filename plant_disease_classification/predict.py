#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf

image_size = 128

session = tf.Session()
saver = tf.train.import_meta_graph('plants-disease-model.meta')
saver.restore(session, tf.train.latest_checkpoint('./'))

def predict(filename):
    image = cv2.imread(filename)
    # Resizing the image to our desired size and
    # preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
