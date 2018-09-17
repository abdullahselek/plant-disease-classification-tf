#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import click
import trainer
import classifier

help_message = '''
  Disease classification on different plants with using Machine Learning and Convolutional Neural Networks.
  Usage
    $ python plant_disease_classification [<options>]
  Options
    --train, -t             Train dataset and create model
    --classify, -c          Classify image from given file path, train your dataset before classification
    --help, -h              Display help message
  Examples
    $ python plant_disease_classification --train
    $ python plant_disease_classification --classify
    $ python plant_disease_classification --help
'''

@click.command(add_help_option=False)
@click.option('-t', '--train', is_flag=True, default=False, help='Train dataset and create model')
@click.option('-c', '--classify', is_flag=True, default=False, help='Classify image from given file path')
@click.option('-h', '--help', is_flag=True, default=False, help='Display help message')

def main(train, classify, help):
    if (help):
        print(help_message)
        sys.exit(0)
    else:
        if (train):
            iteration = click.prompt('Iteration count for training model', type=int)
            trainer.train(num_iteration=iteration)
        else:
            image_file_path = click.prompt('Image file path that is going to be classified', type=str)
            classifier.classify(file_path=image_file_path)
                  

if __name__ == '__main__':
    main()
