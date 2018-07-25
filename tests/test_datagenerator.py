#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from plant_disease_detection import datagenerator

class DataGeneratorTest(unittest.TestCase):

    def test_find_images(self):
        image_paths = datagenerator.find_images('datasets/train')
        self.assertEqual(len(image_paths), 21917)

    def test_load_train_data(self):
        train_path = os.path.join('plant_disease_detection', 'datasets/train')
        classes = os.listdir(train_path)
        images, labels, img_names, class_array = datagenerator.load_train_data(train_path, 128, classes)
        self.assertEqual(len(images), 21917)
        self.assertEqual(len(labels), 21917)
        self.assertEqual(len(img_names), 21917)
        self.assertEqual(len(class_array), 21917)
