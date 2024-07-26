import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import scipy.stats as sc_stats

class StdDevStrategy(Strategy):
    def init(self):
        self.daily_changes = self.data.Close - self.data.Open
        self.daily_avg = self.daily_changes.mean()
        self.daily_std = self.daily_changes.std()

    def next(self):
        daily_change = self.data.Close[-1] - self.data.Open[-1]
        if daily_change > self.daily_avg + 2 * self.daily_std:
            self.buy()
        elif daily_change < self.daily_avg - 2 * self.daily_std:
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
    ticker = input("Enter the ticker symbol: ").upper()
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    start = input(f"Enter the start date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    end = input(f"Enter the end date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, StdDevStrategy, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    
    bt.plot()
    
    # Additional analysis
    data['Daily_Price_Difference'] = data['Close'] - data['Open']
    daily_std = data['Daily_Price_Difference'].std()
    daily_avg = data['Daily_Price_Difference'].mean()

    current_stock_price = data['Close'].iloc[-1]
    prices_data = {
        'Frequency': ['Daily'],
        '1st Std Deviation (-)': [current_stock_price - daily_avg - daily_std],
        '1st Std Deviation (+)': [current_stock_price - daily_avg + daily_std],
        '2nd Std Deviation (-)': [current_stock_price - daily_avg - 2 * daily_std],
        '2nd Std Deviation (+)': [current_stock_price - daily_avg + 2 * daily_std],
        '3rd Std Deviation (-)': [current_stock_price - daily_avg - 3 * daily_std],
        '3rd Std Deviation (+)': [current_stock_price - daily_avg + 3 * daily_std]
    }
    prices_table = pd.DataFrame(prices_data)

    print("Standard Deviations:")
    print(prices_table)

    plt.figure(figsize=(8, 6))
    plt.hist(data['Daily_Price_Difference'], bins=30, color='blue', alpha=0.4, label='Daily')
    plt.title(f'{ticker} Daily Price Difference Histogram')
    plt.xlabel('Price Difference')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    # Generate and plot distribution fits for daily price differences
    mean_change = data['Daily_Price_Difference'].mean()
    plt.figure(figsize=(10, 6))
    hist_data = plt.hist(data['Daily_Price_Difference'], bins=30, color='blue', alpha=0.5, density=True, label='Daily Price Difference')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = sc_stats.norm.pdf(x, mean_change, daily_std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
    plt.title(f'Normal Distribution Fit for Daily Price Differences of {ticker}')
    plt.xlabel('Daily Price Difference')
    plt.ylabel('Density')
    plt.axvline(mean_change, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(mean_change + daily_std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.axvline(mean_change - daily_std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    plt.axvline(mean_change + 2 * daily_std, color='yellow', linestyle='dashed', linewidth=2, label='+2 STD')
    plt.axvline(mean_change - 2 * daily_std, color='yellow', linestyle='dashed', linewidth=2, label='-2 STD')
    plt.axvline(mean_change + 3 * daily_std, color='orange', linestyle='dashed', linewidth=2, label='+3 STD')
    plt.axvline(mean_change - 3 * daily_std, color='orange', linestyle='dashed', linewidth=2, label='-3 STD')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
