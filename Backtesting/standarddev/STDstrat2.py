import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt

class StdDevStrategy(Strategy):
    def init(self):
        self.daily_changes = self.data.Close - self.data.Open

    def next(self):
        if len(self.daily_changes) > 0:
            daily_avg = np.mean(self.daily_changes[:len(self.data)])
            daily_std = np.std(self.daily_changes[:len(self.data)])
            
            daily_change = self.data.Close[-1] - self.data.Open[-1]
            if daily_change > daily_avg + 2 * daily_std:
                self.buy()
            elif daily_change < daily_avg - 2 * daily_std:
                self.sell()
        else:
            # Handle the case where there is not enough data
            pass

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

def main():
    ticker = input("Enter the ticker symbol: ").upper()
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    start = input(f"Enter the start date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    end = input(f"Enter the end date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, StdDevStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    
    bt.plot()

if __name__ == "__main__":
    main()