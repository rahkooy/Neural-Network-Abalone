# Third Neural Network model with with dense layers of depth twice Model 1 and Dropout and Normalisation after each dense layer

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredLogarithmicError

class NeuralNetworkModel3:
    def __init__(self, input_shape):
        # initialise the model
        self.network = keras.Sequential()
        
        # Build the model structure
        self.network.add(layers.Dense(1024, activation='relu', input_shape=(input_shape,)))
        self.network.add(layers.Dense(1024, activation='relu', input_shape=[11])),
        self.network.add(layers.Dropout(0.3)),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(1024, activation='relu')),
        self.network.add(layers.Dropout(0.3)),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(1024, activation='relu')),
        self.network.add(layers.Dropout(0.3)),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(1)),

        

    def compile_model3(self, optimizer='adam', loss = MeanSquaredLogarithmicError(), metrics=['mae']):
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

    def summary3(self):
        #print model summary
        self.network.summary()
