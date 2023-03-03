import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox

import yfinance as yf


class application:
    def __init__(self):
        self.font = ("Arial", 16)
        self.root = tk.Tk()
        self.root.config(bg="#26242f")
        self.root.geometry("225x200")
        self.root.title("Crypto Price")
        self.root.resizable(False, False)

        self.label = tk.Label(
            self.root,
            text="Price:",
            font=self.font,
            justify="center",
            bg="#26242f",
            fg="#FFFFFF",
        )
        self.label.grid(column=0, row=2, padx=2, columnspan=2)

        self.BTC_button = tk.Button(
            self.root,
            text="BTC-USD",
            font=self.font,
            command=self.get_BTC,
            bg="#26242f",
            fg="white",
        )
        self.BTC_button.grid(column=0, row=0, padx=2, pady=3)

        self.ETH_button = tk.Button(
            self.root,
            text="ETH-USD",
            font=self.font,
            command=self.get_ETH,
            bg="#26242f",
            fg="white",
        )
        self.ETH_button.grid(column=1, row=0, padx=2, pady=3)

        self.root.mainloop()

    def get_BTC(self):
        try:
            self.ETH_label.destroy()
            self.pointer.destroy()
        except:
            pass

        self.data = yf.download("BTC-USD", interval="1m")["Close"]
        self.BTC_label = tk.Label(
            self.root,
            text=round(self.data[-1], 2),
            font=("Arial", 26),
            justify="center",
            bg="white",
            bd=10,
        )
        self.BTC_label.grid(column=0, row=4, padx=2, pady=15, columnspan=2)
        self.root.after(60000, self.get_BTC)

        self.pointer = tk.Label(
            self.root,
            text="^",
            font=("Arial", 15),
            bg="#26242f",
            fg="white",
        )
        self.pointer.grid(column=0, row=1, pady=0)

    def get_ETH(self):
        try:
            self.BTC_label.destroy()
            self.pointer.destroy()
        except:
            pass

        self.data = yf.download("ETH-USD", interval="1m")["Close"]
        self.ETH_label = tk.Label(
            self.root,
            text=round(self.data[-1], 2),
            font=("Arial", 26),
            justify="center",
            bg="white",
            bd=10,
        )
        self.ETH_label.grid(column=0, row=4, padx=10, pady=15, columnspan=2)
        self.root.after(60000, self.get_ETH)

        self.pointer = tk.Label(
            self.root,
            text="^",
            font=("Arial", 15),
            bg="#26242f",
            fg="white",
        )
        self.pointer.grid(column=1, row=1, pady=0)


application()
