#!/usr/bin/env python

'''Disease classification on different plants with Neural Networks.'''

from __future__ import absolute_import

__author__       = 'Abdullah Selek'
__email__        = 'abdullahselek@gmail.com'
__copyright__    = 'Copyright (c) 2018 Abdullah Selek'
__license__      = 'MIT License'
__version__      = '0.1'
__url__          = 'https://github.com/abdullahselek/plant-disease-classification'
__download_url__ = 'https://github.com/abdullahselek/plant-disease-classification'
__description__  = 'Disease classification on different plants with Neural Networks.'

from plant_disease_classification import (
    datagenerator,
    classifier
)

from plant_disease_classification.dataset import DataSet
