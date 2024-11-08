# Second Neural Network model only with dense layers of depth twice Model 1

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredLogarithmicError

class NeuralNetworkModel2:
    def __init__(self, input_shape):
        # initialise the model
        self.network = keras.Sequential()
        
        # Build the model structure
        self.network.add(layers.Dense(512, activation='relu', input_shape=(input_shape,)))
        self.network.add(layers.Dense(512, activation='relu'))
        self.network.add(layers.Dense(512, activation='relu'))
        self.network.add(layers.Dense(1))

    def compile_model2(self, optimizer='adam', loss = MeanSquaredLogarithmicError(), metrics=['mae']):
        # compile the model
        self.network.compile(optimizer =optimizer, loss=loss, metrics=metrics)

    def fit(self, X_train, y_train, validation_data, batch_size, epochs, callbacks, verbose=0):
        return self.network.fit(
            X_train, y_train,
            validation_data=validation_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=verbose
        )

    def summary2(self):
        #print model summary
        self.network.summary()
