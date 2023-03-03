import streamlit as st
import yfinance as yf
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly.subplots import make_subplots
from plotly import graph_objs as go

# streamlit run "Python\Forecasting Dashboard\dashboard.py"


class homepage:
    def __init__(self) -> None:
        st.set_page_config(layout="wide")
        st.title("Homepage")

        self.market_option = st.selectbox(
            label="Select Market", options=["Stocks", "Crypto"]
        )

        self.market_button = st.button(
            label="Go",
            on_click=self.direction,
        )

    def direction(self):
        if self.market_option == "Stocks":
            pass
        if self.market_option == "Crypto":
            pass


homepage()
