# Neural Network Models for Abalone biology dataset

## Project Purpose
- Apply pre-trained models for the predicting age in Abalone, a 
[Kaggle competition](www.kaggle.com/competitions/playground-series-s4e4/overview/$citation)
Walter Reade and Ashley Chow. Regression with an Abalone Dataset. https://kaggle.com/competitions/playground-series-s4e4, 2024. Kaggle.
- Implement new Neural Network models for predicting age in Abalone
- Compare the performance of the models


## Dataset
- [Dataset at the UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone), 
[DOI](doi.org/10.24432/C55C7W)
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0) 
- the Potential  main paper


## Requirements
- `numpy`
- `seaborn`
- `pyplot` from `matplotlib`
- `tensorflow`
- `keras`
  - Optimizer: `Adam`
  - Models: `InceptionV3`, `ResNet50`, `DenseNet`
- `sklearn`
  - Metrics: `confusion_matrix`, `accuracy_score`

## Utils Directory
The `utils` directory includes Python files, each of which contains a CNN model and possibly some variants of it. It also contains helper.py, which includes the following auxiliary functions:
- `datagen_train_dir`
- `datagen_test_dir`
- `plot_history`
- `plot_confusion_matrix`
- `plot_roc_au`

## Models Considered and Their Performance
1. **Our Model**: 
   - ...
   - ... accuracy