import yfinance as yf
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import numpy as np

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

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data

def get_available_dates(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    return hist.index.min(), hist.index.max()

def calculate_annualized_return(data):
    # Calculate the total return from start to end
    total_return = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1
    
    # Calculate the number of days in the period
    days = (data.index[-1] - data.index[0]).days
    
    # Annualize the return
    annualized_return = (1 + total_return) ** (365 / days) - 1
    return annualized_return

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
    
    # Calculate and print the annualized Buy and Hold return
    buy_and_hold_annualized_return = calculate_annualized_return(data)
    print(f"Annualized Buy and Hold Return: {buy_and_hold_annualized_return * 100:.2f}%")
    
    bt.plot()

if __name__ == "__main__":
    main()
