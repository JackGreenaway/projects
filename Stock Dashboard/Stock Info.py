import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime
import time


class stock_info:
    def __init__(self) -> None:
        st.set_page_config(layout="wide")

        st.sidebar.title("Stock Information")
        self.ticker = st.sidebar.text_input(
            label="Select a stock", placeholder="Ticker", max_chars=5, value="NVDA"
        )

        self.news_button = st.sidebar.checkbox(
            label="Show Sky News",
            # on_click=self.news_channel,
        )

        self.fetch_button = st.sidebar.button(
            label="Get Data",
            on_click=self.get_data,
        )

    def get_data(self):
        try:
            self.df = yf.download(self.ticker, period="1y")
            time = datetime.now()
            time_str = time.strftime("%d/%m/%Y @ %H:%M:%S")
            st.title(f"Current {self.ticker} data")
            st.subheader(time_str)
            if self.df.empty:
                st.header(f"Incorrect ticker ({self.ticker}) or no data available")
            else:
                # st.subheader("Raw Data")
                # st.dataframe(self.df[::-1], height=210)
                pass
        except:
            st.header("Enter a ticker")
        self.present_data()
        if self.news_button:
            self.news_channel()

    def present_data(self):
        self.df["pc_close"] = self.df["Close"].pct_change()
        self.df["pc_open"] = self.df["Open"].pct_change()
        self.df["pc_volume"] = self.df["Volume"].pct_change()

        col1, col2, col3 = st.columns(3)

        col1.header("Current Price")
        col1.metric(
            label="",
            # label="Current Price",
            value="%.2f" % round(self.df["Close"][::-1][0], 2),
            delta="%.2f" % (round(self.df["pc_close"][::-1][0], 4) * 100),
        )

        col2.header("Open Price")
        col2.metric(
            label="",
            value="%.2f" % round(self.df["Open"][::-1][0], 2),
            delta="%.2f" % (round(self.df["pc_open"][::-1][0], 4) * 100),
        )

        col3.header("Volume")
        col3.metric(
            label="",
            value=round(self.df["Volume"][::-1][0], 2),
            delta="%.2f" % (round(self.df["pc_volume"][::-1][0], 4) * 100),
        )
        # while True:
        #    time.sleep(60)
        #    self.present_data()

    def news_channel(self):
        st.video("https://youtu.be/9Auq9mYxFEE")


stock_info()
