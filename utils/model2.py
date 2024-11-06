# Second Neural Network model only with dense layers of depth twice model1


from tensorflow import keras
from tensorflow.keras import layers

model2 = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[4]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])
