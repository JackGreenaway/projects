import streamlit as st
import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly.subplots import make_subplots
from plotly import graph_objs as go


class dashboard:
    def __init__(self) -> None:
        # st.title("Stock Dashboard")
        # st.sidebar.write("Sidebar")
        st.set_page_config(layout="wide")

        st.sidebar.title("Stock Information")
        self.ticker = st.sidebar.text_input(
            label="Select a stock", placeholder="Ticker", max_chars=5, value="AAPL"
        )

        self.years_data = st.sidebar.select_slider(
            label="Years of data", options=range(11), value=5
        )

        st.sidebar.header("Forecasting")
        self.num_years = st.sidebar.select_slider(
            label="Number of years forecasted", options=range(6), value=1
        )

        # self.forecast_button = st.sidebar.button(
        #    label="Forecast Data",
        #    on_click=self.forecast_data,
        # )

        self.fetch_button = st.sidebar.button(
            label="Get Data",
            on_click=self.get_data,
        )

    def get_data(self):
        st.title(f"{self.ticker} Data and Forecast")
        years = str(self.years_data) + "y"
        self.df = yf.download(self.ticker, period=years)
        if self.df.empty:
            st.header(f"Incorrect ticker ({self.ticker}) or no data available")
        else:
            st.subheader("Raw Data")
            st.dataframe(self.df[::-1], height=210)
            self.plot_raw_data()

    @st.cache(suppress_st_warning=True)
    def plot_raw_data(self):
        fig_raw = make_subplots(specs=[[{"secondary_y": True}]])
        fig_raw.add_trace(
            go.Scatter(
                x=self.df.index,
                y=self.df["Adj Close"],
                name="Historical Close Price ($)",
            )
        )
        fig_raw.add_trace(
            go.Bar(
                x=self.df.index,
                y=self.df["Volume"],
                name="Volume",
                opacity=0.3,
            ),
            secondary_y=True,
        )
        fig_raw.layout.update(
            title_text="Time Series Data",
            xaxis_rangeslider_visible=True,
            xaxis_title="Date",
            # yaxis_title="Adj Close ($)",
            height=600,
            width=900,
        )
        st.plotly_chart(fig_raw)
        self.forecast_data()

    def forecast_data(self):
        df_prophet = self.df.tz_convert(None)
        df_prophet = df_prophet.reset_index()
        df_train = df_prophet[["Date", "Adj Close"]]
        # df_train = df_train["Date"].dt.tz_convert(None)
        df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})

        periods = self.num_years * 365

        self.m = Prophet()
        self.m.fit(df_train)
        future = self.m.make_future_dataframe(periods=periods)
        self.forecast = self.m.predict(future)

        st.subheader("Forecasted Data")
        st.dataframe(self.forecast[len(self.df) :], height=210)

        self.forecast_plot()

    def forecast_plot(self):
        fig_forecast = plot_plotly(self.m, self.forecast)
        st.plotly_chart(fig_forecast)
        fig_forecast.layout.update(
            title_text="Time Series Data",
            xaxis_rangeslider_visible=True,
            xaxis_title="Date",
            yaxis_title="Adj Close ($)",
        )


dashboard()
