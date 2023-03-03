import yfinance as yf
import streamlit as st


class stock_info:
    def __init__(self) -> None:
        st.set_page_config(layout="wide")

        st.sidebar.title("Stock Information")
        self.ticker = st.sidebar.text_input(
            label="Select a stock",
            placeholder="Ticker",
            max_chars=5,
        )

        self.fetch_button = st.sidebar.button(
            label="Get Data",
            on_click=self.get_data,
        )

    def get_data(self):
        st.write("This does nothing right now")
        pass


stock_info()
