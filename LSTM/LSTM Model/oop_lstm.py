import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense, LeakyReLU


class lstm_model:
    def __init__(
        self,
        ticker,
    ):
        self.ticker = ticker

        # methods
        self.load_data()
        self.data_scale()
        self.preprocessing()
        self.build_model()
        self.train_model()
        self.predictions_evaluation()

    def load_data(self):
        self.raw_data = yf.download(self.ticker, start="2018-01-01")["Adj Close"]

    def data_scale(self):
        # scale the dataset
        data_array = np.array(self.raw_data)
        data_array = data_array.reshape(-1, 1)

        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(data_array)

    def preprocessing(self, lookback=10):
        # split data into features (x, y)
        x, y = [], []
        for i in range(len(self.scaled_data) - lookback):
            x.append(self.scaled_data[i : i + lookback])
            y.append(self.scaled_data[i + lookback])
        x, y = np.array(x), np.array(y)

        x = x.reshape(x.shape[0], x.shape[1], 1)
        y = y.reshape(-1, 1)

        # train/test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False, random_state=42
        )

    def check_train_test(self):
        print(
            self.x_train.shape, self.x_test.shape, self.y_train.shape, self.y_test.shape
        )

    def build_model(self):
        model = Sequential()
        model.add(
            LSTM(
                30,
                return_sequences=True,
                activation="relu",
                input_shape=(self.x_train.shape[1], self.x_test.shape[2]),
            )
        )
        model.add(LeakyReLU(alpha=0.3))
        model.add(LSTM(15, return_sequences=False))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(10))

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mean_squared_error",
        )

        self.model = model
        return self.model

    def train_model(self):
        self.trained_model = self.model.fit(
            self.x_train, self.y_train, epochs=30, batch_size=32, validation_split=0.2
        )

    def model_evaluation(self):
        model = self.trained_model
        plt.plot(model.history["loss"], label="Training loss")
        plt.plot(model.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    def predictions_evaluation(self):
        predicitonsLSTM = self.model.predict(self.x_test)
        predicitonsLSTM = self.scaler.inverse_transform(predicitonsLSTM)

        y_test_descaled = self.scaler.inverse_transform(self.y_test)

        # print(predicitonsLSTM)

        plt.plot(predicitonsLSTM)
        plt.plot(y_test_descaled)

        plt.title(f"{self.ticker} Forecast")
        plt.ylabel("Price ($)")
        plt.xlabel("Date")

        legend = ["Predicted", "Historical"]
        plt.legend(legend)
        plt.show()

        self.predictionsLSTM = predicitonsLSTM
        self.y_test_descaled = y_test_descaled

    def future_predict(self, days=10):
        predictions = []
        x_pred = self.x_test[-1:, :, :]
        y_pred = self.y_test[-1]

        for i in range(days):
            x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)
            y_pred = self.model.predict(x_pred)
            predictions.append(y_pred.flatten()[0])

        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))

        # print(predictions)

        self.predictions = predictions

        self.plot_predictions()

    def plot_predictions(self, size=30):
        temp = np.concatenate((self.predictionsLSTM, self.predictions))
        plt.plot(temp)
        plt.plot(self.y_test_descaled)

        labels = ["Predicted", "Historical"]
        plt.legend(labels)
        print(f"{'-' * 100}\n")
        print(f"Predictions for next {size} days")
        print(self.predictions)
        plt.show()


instance = lstm_model("NVDA")

instance.future_predict()
