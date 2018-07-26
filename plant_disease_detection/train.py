#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from plant_disease_detection import datagenerator

def get_data():
    train_path = os.path.join('plant_disease_detection', 'datasets/train')
    classes = os.listdir(train_path)

    # 20% of the data will automatically be used for validation
    validation_size = 0.2
    img_size = 128

    # Load and read training data
    data = datagenerator.read_train_sets(train_path, img_size, classes, validation_size)
    print('Complete reading input data. Will Now print a snippet of it')
    print('Number of files in Training-set:\t\t{}'.format(len(data.train.labels)))
    print('Number of files in Validation-set:\t{}'.format(len(data.valid.labels)))
    return data
