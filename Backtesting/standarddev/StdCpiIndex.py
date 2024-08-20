import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

class MultiIndexStdDevStrategy(Strategy):
    def init(self):
        # Define the indices and ETFs we want to track
        self.tickers = ['QQQ']
        
        # Fetch data for all tickers
        self.all_data = {}
        for ticker in self.tickers:
            self.all_data[ticker] = yf.download(ticker, start=self.data.index[0], end=self.data.index[-1])
        
        # Calculate daily changes for all tickers
        self.daily_changes = {}
        for ticker in self.tickers:
            self.daily_changes[ticker] = self.all_data[ticker]['Close'] - self.all_data[ticker]['Open']
            
        # Calculate mean and standard deviation for all tickers
        self.daily_avg = {}
        self.daily_std = {}
        for ticker in self.tickers:
            self.daily_avg[ticker] = np.mean(self.daily_changes[ticker])
            self.daily_std[ticker] = np.std(self.daily_changes[ticker])

    def next(self):
        if len(self.data) > 1:  # Ensure we have at least two data points
            negative_std_count = 0
            
            for ticker in self.tickers:
                latest_date = self.data.index[-1]
                if latest_date in self.all_data[ticker].index:
                    daily_change = self.all_data[ticker].loc[latest_date, 'Close'] - self.all_data[ticker].loc[latest_date, 'Open']
                    
                    if daily_change < self.daily_avg[ticker] - 2 * self.daily_std[ticker]:
                        negative_std_count += 1
            
            # If all indices show negative 2 std dev move, generate buy signal
            if negative_std_count == len(self.tickers):
                if not self.position:
                    self.buy()
            else:
                if self.position:
                    self.sell()

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data

def get_available_dates(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    return hist.index.min(), hist.index.max()

def main():
    ticker = 'SPY'  # We'll use SPY as our main trading instrument
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    start = input(f"Enter the start date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    end = input(f"Enter the end date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, MultiIndexStdDevStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    
    bt.plot()

if __name__ == "__main__":
    main()
