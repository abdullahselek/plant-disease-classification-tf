#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class DataSet(object):

    def __init__(self, *args):
        if len(args) > 3:
            self._num_examples = args[0].shape[0]
            self._images = args[0]
            self._labels = args[1]
            self._img_names = args[2]
            self._cls = args[3]
        self._epochs_done = 0
        self._index_in_epoch = 0
        self.train = None
        self.valid = None

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set.
        Args:
          batch_size (int):
            Size of batch.
        Returns:
           Tuples of images, labels, image names and class. 
        """

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]
