import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import messagebox

import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class application:
    def __init__(self):

        self.font = ("Arial", 16)
        self.root = tk.Tk()
        self.root.config(bg="#26242f")

        self.menubar = tk.Menu(self.root, bg="#32242F")

        self.filemenu = tk.Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="Restart", command=self.restart)
        self.filemenu.add_command(label="Exit", command=self.exit_app)

        self.propmenu = tk.Menu(self.menubar, tearoff=0)
        self.propmenu.add_command(label="Live Price", command=self.live_price)

        self.menubar.add_cascade(menu=self.filemenu, label="File")
        self.menubar.add_cascade(menu=self.propmenu, label="Properties")
        self.root.config(menu=self.menubar)

        self.root.geometry("425x620")
        self.root.title("Stock Information")
        self.root.resizable(False, False)

        self.label = tk.Label(
            self.root,
            text="Ticker:",
            font=self.font,
            justify="center",
            bg="#26242f",
            fg="#FFFFFF",
        )
        self.label.grid(column=0, row=0, padx=2)

        self.textbox = tk.Text(self.root, height=1, font=self.font, width=12)
        self.textbox.bind("<KeyPress>", self.shortcut)
        self.textbox.grid(column=1, row=0, padx=2, pady=3)

        self.check_state = tk.IntVar()
        self.check = tk.Checkbutton(  # change to a switch
            self.root,
            text="Show Graph",
            font=self.font,
            variable=self.check_state,
            bg="#26242f",
            fg="white",
            selectcolor="#26242f",
        )
        self.check.grid(column=2, row=0, padx=2, pady=3)

        self.button = tk.Button(
            self.root,
            text="Go",
            font=self.font,
            command=self.get_data,
            bg="#26242f",
            fg="white",
        )
        self.button.grid(column=3, row=0, padx=2, pady=3)

        self.root.mainloop()

    def get_data(self):
        try:
            self.error_mes.destroy()
        except:
            pass

        ticker = self.textbox.get("1.0", tk.END)
        ticker = ticker.upper()
        self.data = yf.download(ticker, period="1y")[["Open", "Close", "Volume"]]
        if self.data.empty:
            self.no_data()
        else:
            frame = Frame(self.root)
            frame.grid(column=0, row=1, columnspan=4, pady=10)

            style = ttk.Style()
            style.configure("Treeview.Heading", font=("Arial", 12))

            # columns = []
            # columns.append('Date')
            # for i in self.data.columns:
            #    columns.append(i)

            columns = self.data.columns.tolist()
            self.tree = ttk.Treeview(
                frame, columns=columns, show="headings", height="5"
            )

            length = 100
            for heading in columns:
                heading = str(heading)
                self.tree.column(heading, width=130, anchor="center")
                self.tree.heading(heading, text=heading)
            for index, row in self.data[::-1].head(length).iterrows():
                self.tree.insert("", tk.END, text=index, values=list(round(row, 2)))

            # for date in self.data[::-1].head(length).index.strftime('%d/%m/%Y'):
            #    self.tree.insert('', tk.END, text=index, values=date)

            self.tree.pack()

        if self.check_state.get() == 1:
            self.plot_data()
        else:
            pass

    def plot_data(self):
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111).plot(self.data.Close[-30:])
        # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=2, hspace=0)
        plt.title("t-30 Close Price ($)")
        plt.xticks(rotation=20)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig)
        canvas.get_tk_widget().grid(column=0, row=2, columnspan=4)
        canvas.draw()

    def no_data(self):
        self.error_mes = tk.Label(
            self.root,
            text="Incorrect ticker or no data available",
            font=("", 15, "bold"),
        )
        self.error_mes.grid(column=0, row=1)

    def live_price(self):
        try:
            cur_price = self.data.Close[-1]
            messagebox.showinfo("Current Price", f"Current price: {round(cur_price,2)}")
        except:
            messagebox.showinfo("Error 😲", "Please input a ticker")

    def shortcut(self, event):
        if event.keysym == "Return" and event.state == 1:
            self.get_data()
        # print(event.keysym)
        # print(event.state)

    def restart(self):
        self.root.destroy()
        application()

    def exit_app(self):
        self.root.destroy()


application()
