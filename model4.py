# Fourth Neural Network model with much more layers
# https://www.kaggle.com/code/eneskosar19/abalone-age-prediction-ann-regression#MODEL


from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    #layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dense(units=256, activation='relu')
    layers.BatchNormalization()
    layers.Dropout(0,1)
    layers.Dense(units=256, activation='relu')
    layers.BatchNormalization()
    layers.Dropout(0,1)
    layers.Dense(units=128, activation='relu')
    layers.BatchNormalization()
    layers.Dropout(0,1)
    layers.Dense(units=64, activation='relu')
    layers.BatchNormalization()

    layers.Dense(units=32, activation='relu')
    layers.BatchNormalization()

    layers.Dense(units=16, activation='relu')
    layers.BatchNormalization()

    layers.Dense(units=8, activation='relu')
    layers.BatchNormalization()

    layers.Dense(units=3, activation='relu')
    # Output layer
    layers.Dense(units=1, activation='linear')
])