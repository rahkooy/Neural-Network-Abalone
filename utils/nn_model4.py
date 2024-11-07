# Fourth Neural Network model with much more layers
# https://www.kaggle.com/code/eneskosar19/abalone-age-prediction-ann-regression#MODEL


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredLogarithmicError

class NeuralNetworkModel4:
    def __init__(self):
        self.network = keras.Sequential()
        self.network.add(layers.Dense(1024, activation='relu', input_shape=[7])),
        self.network.add(layers.Dense(256, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dropout(0,1)),
        self.network.add(layers.Dense(256, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dropout(0,1)),
        self.network.add(layers.Dense(128, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dropout(0,1)),
        self.network.add(layers.Dense(64, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(32, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(16, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(8, activation='relu')),
        self.network.add(layers.BatchNormalization()),
        self.network.add(layers.Dense(3, activation='relu')),
        # Output layer
        self.network.add(layers.Dense(1, activation='linear'))

    def compile_model4(self, optimizer='adam', loss = MeanSquaredLogarithmicError(), metrics=['mae']):
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

    def summary4(self):
        #print model summary
        self.network.summary()
