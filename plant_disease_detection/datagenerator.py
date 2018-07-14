#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

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
