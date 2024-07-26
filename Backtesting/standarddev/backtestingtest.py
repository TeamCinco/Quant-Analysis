import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import matplotlib.pyplot as plt

class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)  # Short-term moving average
        self.ma2 = self.I(SMA, price, 20)  # Long-term moving average

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()  # Buy signal when short-term MA crosses above long-term MA
        elif crossover(self.ma2, self.ma1):
            self.sell()  # Sell signal when long-term MA crosses above short-term MA

# Function to fetch data using yfinance
def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data

# Function to get available date ranges
def get_available_dates(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    return hist.index.min(), hist.index.max()

# Main function
def main():
    ticker = input("Enter the ticker symbol: ").upper()
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    start = input(f"Enter the start date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    end = input(f"Enter the end date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, SmaCross, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    
    bt.plot()

if __name__ == "__main__":
    main()
