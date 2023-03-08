import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tools.eval_measures import rmse
import os

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
        # load the raw data
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
        # for debugging to check the shape of the data
        print(
            self.x_train.shape, self.x_test.shape, self.y_train.shape, self.y_test.shape
        )

    def build_model(self):
        # define a model that we will train the data on
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
        # train the model on the historical train data
        self.trained_model = self.model.fit(
            self.x_train, self.y_train, epochs=50, batch_size=32, validation_split=0.2
        )

    def model_evaluation(self):
        # evaluate and plot the model to check its training and check for over/underfitting
        model = self.trained_model
        plt.plot(model.history["loss"], label="Training loss")
        plt.plot(model.history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()

    def predictions_evaluation(self):
        # predict the test data
        model_historical_pred = self.model.predict(self.x_test)
        model_historical_pred = self.scaler.inverse_transform(model_historical_pred)

        os.system("cls")
        print("Training Complete")

        # descale the data for plotting
        y_test_descaled = self.scaler.inverse_transform(self.y_test)

        # add error metrics below
        root_mse = rmse(model_historical_pred.flatten(), y_test_descaled.flatten())
        mean_se = root_mse**2
        his_mean = y_test_descaled.mean()
        pred_mean = model_historical_pred.mean()
        print(f"{'---'*15}")
        print("Error Metrics (Historical vs Model Predicted)")
        print(f"RMSE: {round(root_mse,2)}")
        print(f"MSE: {round(mean_se,2)}")
        print(
            f"Historical Mean: {round(his_mean,2)} | Predicted Mean: {round(pred_mean,2)} | Difference {round((his_mean - pred_mean),2)}\n"
        )

        self.model_historical_pred = model_historical_pred
        self.y_test_descaled = y_test_descaled

    def predictions_evaluation_plot(self):
        # plot an evaluation
        plt.title("Historical vs Model Predicted")
        plt.plot(self.model_historical_pred)
        plt.plot(self.y_test_descaled)

        plt.title(f"{self.ticker} Forecast")
        plt.ylabel("Price ($)")
        plt.xlabel("Date")

        legend = ["Predicted", "Historical"]
        plt.legend(legend)
        plt.show()

    def future_predict(self, days=10):
        # create a list to store the future predictions
        predictions = []
        x_pred = self.x_test[-1:, :, :]
        y_pred = self.y_test[-1]

        # loop over the days predicted for, predicting each day one at a time and adding the new predictions to the dataset to fuel more predictions
        for i in range(days):
            x_pred = np.append(x_pred[:, 1:, :], y_pred.reshape(1, 1, 1), axis=1)
            y_pred = self.model.predict(x_pred)
            predictions.append(y_pred.flatten()[0])

        # format the predictions
        predictions = np.array(predictions)
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1))

        print(f"\n{'---'*15}")
        print(f"Future Predictions:\n {predictions}\n")

        self.predictions = predictions
        self.days = days

        self.future_predict_plot()

    def future_predict_plot(self):
        # get the datetime for today
        date_today = datetime.now()

        # create two dataframes to store the dates relevant to the data
        # future dates
        dates_future = pd.date_range(
            date_today, date_today + timedelta(len(self.predictions) - 1), freq="D"
        )
        # historical dates
        dates_historical = pd.date_range(
            date_today - timedelta(len(self.y_test_descaled) - 1), date_today, freq="D"
        )

        # add the future and historical data to their respective dataframes
        future_df = pd.DataFrame(
            {"date": dates_future, "future_pred": self.predictions.flatten()}
        )
        future_df = future_df.set_index("date")

        past_df = pd.DataFrame(
            {
                "date": dates_historical,
                "historical": self.y_test_descaled.flatten(),
                "historical_pred": self.model_historical_pred.flatten(),
            }
        )
        past_df = past_df.set_index("date")

        # plot the dataframes visually
        plt.title(f"{self.ticker} Future Forecast of {self.days} days")
        plt.plot(future_df)
        plt.plot(past_df)
        labels = ["Future Predicitons", "Historical", "Historical Predictions"]
        plt.legend(labels)
        plt.show()


instance = lstm_model("AAPL")

instance.future_predict()
