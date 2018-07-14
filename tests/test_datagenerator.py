#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from plant_disease_detection import datagenerator

class DataGeneratorTest(unittest.TestCase):

    def test_find_images(self):
        image_paths = datagenerator.find_images('datasets/train')
        self.assertEqual(len(image_paths), 21917)
