#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest

from plant_disease_classification import predict

class PredictTest(unittest.TestCase):

    def test_predict(self):
        predict.predict()
