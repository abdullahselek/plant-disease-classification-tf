#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import click

from plant_disease_classification import (
    datagenerator,
    predict
)

help_message = '''
  Disease classification on different plants with using Machine Learning and Convolutional Neural Networks.
  Usage
    $ python plant_disease_classification [<options>]
  Options
    --help, -h              Display help message
    --predict, -p           Predict image from given file path
    --version, -v           Display installed version
  Examples
    $ python plant_disease_classification --help
    $ python plant_disease_classification --predict
    $ python plant_disease_classification --version
'''
