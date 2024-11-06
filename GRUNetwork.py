import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras
from keras.models import load_model

class GRUNetwork:
    def __init__(self,x_train, x_test, y_train, y_test, eta, batch_size, epochs):
        self.model = Sequential()

        self.model.add(GRU(75, return_sequences=True, input_shape = (x_train.shape[1], x_train.shape[2]), activation ='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(GRU(50, return_sequences=True, activation ='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(GRU(35, return_sequences=False, activation ='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(20))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(1))

        self.optimizer = Adam(learning_rate=eta)
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=int(epochs*0.8), restore_best_weights=True)

        self.model.compile(optimizer=self.optimizer, loss='mean_squared_error', metrics = ['mae'])

    def train(self, x_train, y_train, epochs, batch_size, x_test, y_test):
        self.model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test),callbacks=[self.early_stopping], batch_size=batch_size)

    def evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test)

    def predict(self, x_test, y_test, scaler_y):
        y_pred = self.model.predict(x_test)
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1))
        y_test = scaler_y.inverse_transform(y_test.reshape(-1,1))

        return y_pred, y_test

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = load_model(filename)

    def plot_prediction(self,y_pred, y_test):
        # Figure Size
        plt.figure(figsize=(10, 6))

        # Ploting lines
        plt.plot(y_test, label="Energy Consumption", color='blue', linestyle='dashed')
        plt.plot(y_pred[:, 0], label="Prediction", color='red', alpha=0.7)

        # Title and Label
        plt.title("Real values x Predicted vales", fontsize=14)
        plt.xlabel("Hours", fontsize=12)
        plt.ylabel("Energetic Balance", fontsize=12)

        # Ading legends of lines
        plt.legend()

        # Show Graph
        return plt.show()

__all__ = ['GRUNetwork']