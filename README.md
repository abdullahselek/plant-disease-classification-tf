# plant-disease-classification

A Convolutional Neural Network with TensorFlow and OpenCV using a dataset that contains different leafs of different plants. Repository doesn't have the dataset, you can download dataset from [PlantVillage Disease](https://www.crowdai.org/challenges/1) after registration.

## Requirements

- tensorflow
- numpy
- scikit-learn
- opencv-python
- opencv-contrib-python

## Training

Call **trian.py** with `train` method with training and tests datasets.

```
python plant_disease_classification/train.py --train plant_disease_classification/datasets/train --val plant_disease_classification/datasets/test --num_classes 38
```

<p align="center">
    <img src="https://github.com/abdullahselek/plant-disease-classification/blob/master/screenshot.png"/>
</p>

## Model

After successfull training, model is created under **plant_disease_classification/model/** folder.

## Prediction

Call `predict` function of **predict.py** with selected image file path. Below there is sample command for CLI for default image path.

```
python plant_disease_classification/predict.py --predict
```
