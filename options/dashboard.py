import tkinter as tk


class Dashboard:


    def __init__(self):
        # Initialize Frame ***************************************************
        self.root = tk.Tk()
        self.root.title('Options Trading Dashboard')
        w, h = 800, 500
        self.root.geometry("%dx%d+0+0" % (w, h))
        self.root.resizable(False, False)

        # Parameters *********************************************************
        self.asset_price = tk.Entry(self.root)
        self.asset_price.place(x=100, y=50)

        self.strike_price = tk.Entry(self.root)
        self.strike_price.place(x=100, y=125)

        self.expiration_date = tk.Entry(self.root)
        self.expiration_date.place(x=250, y=50)

        self.volatility = tk.Entry(self.root)
        self.volatility.place(x=250, y=125)

        self.root.mainloop()

Dashboard()
