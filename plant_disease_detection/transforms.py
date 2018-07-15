#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

meanstd = {
   'mean': [ 0.485, 0.456, 0.406 ],
   'std' : [ 0.229, 0.224, 0.225 ]
}

pca = {
    'eigval': torch.Tensor[ 0.2175, 0.0188, 0.0045],
    'eigvec': torch.Tensor[
      [ -0.5675,  0.7192,  0.4009 ],
      [ -0.5808, -0.0045, -0.8140 ],
      [ -0.5836, -0.6948,  0.4203 ]
    ]
}
