import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import scipy.stats as sc_stats

class ImprovedStdDevStrategy(Strategy):
    window = 20
    trend_window = 50

    def init(self):
        self.sma = self.I(self.SMA, self.data.Close, self.window)
        self.std = self.I(self.STD, self.data.Close, self.window)
        self.trend = self.I(self.SMA, self.data.Close, self.trend_window)

    def next(self):
        # Ensure we have enough data
        if len(self.data) < self.trend_window:
            return
        
        daily_change = self.data.Close[-1] - self.data.Open[-1]
        volatility = self.std[-1] / self.data.Close[-1]
        position_size = 1000 / (volatility * self.data.Close[-1])  # Adjust 1000 based on your risk tolerance

        # Ensure the position size is a valid whole number
        position_size = max(1, int(position_size))
        
        for trade in self.trades:
            if trade.is_long:
                if self.data.Low[-1] <= trade.entry_price * 0.95:  # 5% stop loss
                    trade.close()
                elif self.data.High[-1] >= trade.entry_price * 1.1:  # 10% take profit
                    trade.close()
            else:  # Short trade
                if self.data.High[-1] >= trade.entry_price * 1.05:  # 5% stop loss
                    trade.close()
                elif self.data.Low[-1] <= trade.entry_price * 0.9:  # 10% take profit
                    trade.close()

        if self.data.Close[-1] > self.sma[-1] + 1.5 * self.std[-1] and self.data.Close[-1] > self.trend[-1]:
            self.buy(sl=self.data.Close[-1] * 0.95, size=position_size)
        elif self.data.Close[-1] < self.sma[-1] - 1.5 * self.std[-1] and self.data.Close[-1] < self.trend[-1]:
            self.sell(sl=self.data.Close[-1] * 1.05, size=position_size)

    def SMA(self, array, window):
        """Return SMA of `array` with `window` size."""
        sma = np.full_like(array, np.nan)
        sma[window-1:] = np.convolve(array, np.ones(window), 'valid') / window
        return sma

    def STD(self, array, window):
        """Return rolling standard deviation of `array` with `window` size."""
        std = np.full_like(array, np.nan)
        std[window-1:] = np.std(np.lib.stride_tricks.sliding_window_view(array, window), axis=1)
        return std

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data

def get_available_dates(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    return hist.index.min(), hist.index.max()

def validate_dates(data, start, end):
    if start not in data.index:
        start = data.index[data.index.searchsorted(start)]
    if end not in data.index:
        end = data.index[data.index.searchsorted(end) - 1]
    return start, end

def main():
    ticker = input("Enter the ticker symbol: ").upper()
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    # Use the last 5 years of data
    end = end_date
    start = end - pd.DateOffset(years=5)
    
    data = fetch_data(ticker, start, end)
    start, end = validate_dates(data, start, end)
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, ImprovedStdDevStrategy, commission=.002, exclusive_orders=True)
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
    plt.hist(data['Daily_Price_Difference'].dropna(), bins=30, color='blue', alpha=0.4, label='Daily')
    plt.title(f'{ticker} Daily Price Difference Histogram')
    plt.xlabel('Price Difference')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

    # Generate and plot distribution fits for daily price differences
    mean_change = data['Daily_Price_Difference'].mean()
    plt.figure(figsize=(10, 6))
    hist_data = plt.hist(data['Daily_Price_Difference'].dropna(), bins=30, color='blue', alpha=0.5, density=True, label='Daily Price Difference')
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
