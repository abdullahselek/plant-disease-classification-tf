# plant-disease-classification

A Convolutional Neural Network with TensorFlow and OpenCV using a dataset that contains different leafs of different plants. Repository has the dataset under `plant_disease_classification/datasets` folder.

## Requirements

- tensorflow
- numpy
- scikit-learn
- opencv-python
- opencv-contrib-python

## Training

Call **trainer.py** with `train` method with training and tests datasets.

```
python plant_disease_classification/trainer.py --train plant_disease_classification/datasets/train --val plant_disease_classification/datasets/test --num_classes 38
```

<p align="center">
    <img src="https://github.com/abdullahselek/plant-disease-classification/blob/master/screenshot.png"/>
</p>

## Model

After successfull training, model is created under **plant_disease_classification/ckpts/** folder.

## Classification

Call `classify` function of **classifier.py** with selected image file path. Below there is sample command for CLI for default image path.

```
python plant_disease_classification/classifier.py --classify
```
