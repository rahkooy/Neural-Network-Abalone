# Neural Network Models for Age Prediction in Abalones

## Project Purpose
Predicting abalone's age, which is the number of rings on the shell.

This was the topic of a [Kaggle competition](www.kaggle.com/competitions/playground-series-s4e4/overview/$citation). 
We would like to design neural networks for this task and examine 
their performance. Our goal is to find out whether or not complex 
neural networks perform better than regression. To do this, we start 
with simple neural networks with few layers and depth and grow models 
into the ones with more layers (and adding normalisation and dropout) 
with higher depth. Our experiments show that simpler models perform 
better for this task, which suggests using traditional ML methods such 
as regression.
The reference to the Kaggle competition is the following.

Walter Reade and Ashley Chow. Regression with an Abalone Dataset. 
https://kaggle.com/competitions/playground-series-s4e4, 2024. Kaggle.



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
- `sklearn`
- Metrics: `MeanSquaredLogarithmicError`

## Utils Directory
The `utils` directory includes Python files, each of which contains a 
neural network model and possibly some variants of it. 

## Models Considered and Their Performance
**Model 1**: 
  - only with three dense layers
  - minimum validation Mean Squared Logarithmic Error (msle): 0.0345
  - loss and  msle, as well as validation loss and validation msle 
  are very close

**Model 2**:
  - only with dense layers of depth twice Model 1
  - minimum validation Mean Squared Logarithmic Error (msle): 0.0323
  - loss and  msle, as well as validation loss and validation msle 
  are very close

**Model 3**:
  - with dense layers of depth twice Model 1, with a dropout and a 
  normalisation layer after each dense layer
  - minimum validation Mean Squared Logarithmic Error (msle): 0.0922
  - loss and  msle, as well as validation loss and validation msle 
  are very close

**Model 4**:
  - with more layers than Model 3, however, dense layers  have 
  different and decreasing depth; it contains  dropout and 
  normalisation layers but not in the same order as Model 3
  - minimum validation Mean Squared Logarithmic Error (msle): 0.0591
  - loss and  msle, as well as validation loss and validation 
  msle are very close

## Conclusion
For the given data, a simple regression performs better than 
Complex Neural Networks. This is implied by the fact that the 
increase in the number and the depth of the layers in neural networks 
result in higher error values and more CPU time. 
