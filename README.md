# Neural Network Models for Age Prediction in Abalones

## Project Purpose
Predicting abalone's age, which is th enumber os rings on the shell.

This was the topic of a [Kaggle competition](www.kaggle.com/competitions/playground-series-s4e4/overview/$citation) using regression. However, we would like to design simple neural networks for this purpose. The reference to the competetion is

Walter Reade and Ashley Chow. Regression with an Abalone Dataset. https://kaggle.com/competitions/playground-series-s4e4, 2024. Kaggle.



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