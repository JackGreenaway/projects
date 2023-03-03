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
        st.title("Planned to be a webscraper")


homepage()
