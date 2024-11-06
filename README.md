# Neural Network Models for Age Prediction in Abalones

## Project Purpose
Predicting abalone's age, which is the number of rings on the shell.

This was the topic of a [Kaggle competition](www.kaggle.com/competitions/playground-series-s4e4/overview/$citation) using regression. However, we would like to design simple neural networks for this purpose. The reference to the competetion is

Walter Reade and Ashley Chow. Regression with an Abalone Dataset. https://kaggle.com/competitions/playground-series-s4e4, 2024. Kaggle.



## Dataset
- Contains 4177 abalones with 9 features
- Available at the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/1/abalone) with 
the Creative Commons Attribution 4.0 International (CC BY 4.0)  licence.


## Requirements
- `numpy`
- `pandas`
- `figure` and `pyplot` from `matplotlib`
- `tensorflow`
- `keras`
  - Optimizer: `Adam`
  - Models: `InceptionV3`, `ResNet50`, `DenseNet`
- `sklearn`
  - Metrics: `MeanSquaredLogarithmicError`

## Utils Directory
The `utils` directory includes Python files, each of which contains a neural network model and possibly some variants of it. 

## Models Considered and Their Performance
**Model 1**: 
  - only with three dense layers
  - minimum validation Mean Squared Logarithmic Error (msle): 0.033
  - loss and  msle, as well as validation loss and validation msle are very close

**Model 2**:
  - only with dense layers of depth twice Model 1
  - minimum validation Mean Squared Logarithmic Error (msle): 0.034
  - loss and  msle, as well as validation loss and validation msle are very close

**Model 3**:
  - with dense layers of depth twice Model 1, with a dropout and a normalisation layer after each dense layer
  - minimum validation Mean Squared Logarithmic Error (msle): 0.06
  - loss and  msle, as well as validation loss and validation msle are very close

**Model 4**:
  - with more layers that Model 3, however, dense layers  have different and decreasing depth; it contains  dropout and normalisation layers but not the same order as Model 3
  - minimum validation Mean Squared Logarithmic Error (msle): 0.085
  - loss and  msle, as well as validation loss and validation msle are very close
