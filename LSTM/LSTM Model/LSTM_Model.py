import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM


class lstm_model:
    def __init__(self, ticker, lookback) -> None:
        self.ticker = ticker
        self.lookback = lookback

        self.get_data()
        self.train_test_split()

    def get_data(self):
        self.df = yf.download(self.ticker, period="5y")
        self.data_split(self.df)

    def data_split(self, df):
        x, y = [], []
        for i in range(len(df) - self.lookback):
            x.append(df[i : i + self.lookback, 0])
            y.append(df[i + self.lookback, 0])
        return np.array(x), np.array(y)

    def train_test_split(self):
        model_data = self.df.Close
        x, y = self.data_split(model_data, self.df)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, shuffle=False, random_state=4
        )
        print(self.x_train, self.x_test, self.y_train, self.y_test)

    def data_preprocessing(self):
        pass


instance = lstm_model("^GSPC", 10)
instance.train_test_split


# ------------------------------------------------------------------------------------------------------------------------------


import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM


class LSTMPredictor:
    def __init__(self, ticker, n_steps=10, n_features=1, n_epochs=100, batch_size=32):
        self.ticker = ticker
        self.n_steps = n_steps
        self.n_features = n_features
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def download_data(self):
        self.data = yf.download(self.ticker, start="2010-01-01")

    def split_data(self):
        X, y = [], []
        for i in range(len(self.data) - self.n_steps):
            X.append(self.data[i : i + self.n_steps, 0])
            y.append(self.data[i + self.n_steps, 0])
        return np.array(X), np.array(y)

    def fit(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.data.values.reshape(-1, 1))
        X, y = self.split_data(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        model = Sequential()
        model.add(
            LSTM(
                units=50,
                return_sequences=True,
                input_shape=(self.n_steps, self.n_features),
            )
        )
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X, y, epochs=self.n_epochs, batch_size=self.batch_size, verbose=1)
        self.model = model

    def predict(self):
        scaled_data = self.scaler.transform(self.data.values.reshape(-1, 1))
        X = np.array([scaled_data[-self.n_steps :, 0]])
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        yhat = self.model.predict(X)
        return self.scaler.inverse_transform(yhat)
