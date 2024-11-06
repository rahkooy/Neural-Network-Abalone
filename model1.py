# First Neural Network model only with dense layers of little depth

from tensorflow import keras
from tensorflow.keras import layers

model1 = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[X.shape[1]]),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])