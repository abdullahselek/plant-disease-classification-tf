#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import cv2
import numpy as np

def find_images(path):
    """
    Returns an array with all image paths found dir.
    Following extensions are used to filter images: 'jpg', 'png', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'
    Args:
      dir (str):
        Directory that contains images.
    Returns:
      image_paths (array).
    """

    directory = os.path.join('plant_disease_detection', path)
    # Options for the GNU and BSD find command
    extension_list = ['jpg', 'png', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP']
    find_options = str.format('-iname "*.{0}"', extension_list[0])
    for i in range(1, len(extension_list)):
        find_options = str.format('{0} -o -iname "*.{1}"', find_options, extension_list[i])

    # Find all the images using find command
    process = subprocess.Popen([str.format('find -L {0} {1}', directory, find_options),],
                                stdout=subprocess.PIPE,
                                shell=True)
    image_paths = []
    while True:
        line = process.stdout.readline()
        if line != '':
            filename = os.path.basename(line)
            path = os.path.join(os.path.dirname(line), filename)
            image_paths.append(path)
        else:
            break
    return image_paths

def load_train_data(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    class_array = []

    image_paths = find_images('datasets/train')
    print('Going to read training images')
    for path in image_paths:   
        image = cv2.imread(path)
        image = cv2.resize(image,
                        (image_size, image_size), 0,
                        0, cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = np.multiply(image, 1.0 / 255.0)
        images.append(image)
        label = np.zeros(len(classes))
        label[0] = 1.0
        labels.append('')
        file_base = os.path.basename(file)
        img_names.append(file_base)
        class_array.append('')
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    class_array = np.array(class_array)
    return images, labels, img_names, class_array
