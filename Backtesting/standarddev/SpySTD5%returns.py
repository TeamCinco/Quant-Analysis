
import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest

class ShortTermMacroStrategy(Strategy):
    window = 5  # Reduced window for short-term focus
    std_threshold = 1  # Increased threshold for more sensitive signals
    stop_loss = 0.5  # Tightened stop loss for short-term trading
    max_holding_days = 62  # Extended max holding period slightly

    def init(self):
        self.close = self.data.Close

        self.returns = self.I(self.calculate_returns)
        self.returns_mean = self.I(self.calculate_rolling_mean, self.returns, self.window)
        self.returns_std = self.I(self.calculate_rolling_std, self.returns, self.window)

        # Short-term trend indicators
        self.sma5 = self.I(self.calculate_rolling_mean, self.close, 5)
        self.sma10 = self.I(self.calculate_rolling_mean, self.close, 10)

        self.entry_price = 0
        self.days_held = 0

    def calculate_returns(self):
        returns = np.zeros(len(self.close))
        returns[1:] = np.log(self.close[1:] / self.close[:-1])
        return returns

    def calculate_rolling_mean(self, array, window):
        return pd.Series(array).rolling(window).mean().values

    def calculate_rolling_std(self, array, window):
        return pd.Series(array).rolling(window).std().values

    def next(self):
        if len(self.data) < max(self.window, 10):
            return

        current_return = self.returns[-1]
        current_price = self.close[-1]

        # Short-term trend condition
        uptrend = self.sma5[-1] > self.sma10[-1]

        print(f"Date: {self.data.index[-1]}, Price: {current_price:.2f}, Return: {current_return:.4f}")
        print(f"SMA5: {self.sma5[-1]:.2f}, SMA10: {self.sma10[-1]:.2f}, Uptrend: {uptrend}")

        if not self.position:
            if (current_return < self.returns_mean[-1] - self.std_threshold * self.returns_std[-1] and uptrend):
                self.buy(size=0.98)  # Increased position size for short-term focus
                self.entry_price = current_price
                self.days_held = 0
                print(f"BUY signal at {current_price:.2f}")
        else:
            self.days_held += 1

            if current_price < self.entry_price * (1 - self.stop_loss):
                self.position.close()
                print(f"SELL signal (stop loss) at {current_price:.2f}")
            elif self.days_held >= self.max_holding_days:
                self.position.close()
                print(f"SELL signal (max holding period) at {current_price:.2f}")
            elif (current_return > self.returns_mean[-1] + self.std_threshold * self.returns_std[-1] or not uptrend):
                self.position.close()
                print(f"SELL signal (exit conditions) at {current_price:.2f}")

            if current_price > self.entry_price:
                self.entry_price = current_price

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def calculate_annualized_return(data):
    total_return = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1
    days = (data.index[-1] - data.index[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1
    return annualized_return

def main():
    ticker = "SPY"
    start = "2024-01-25"  
    end = "2024-07-26"
    data = fetch_data(ticker, start, end)
    bt = Backtest(data, ShortTermMacroStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    buy_and_hold_annualized_return = calculate_annualized_return(data)
    print(f"Annualized Buy and Hold Return: {buy_and_hold_annualized_return * 100:.2f}%")
    bt.plot()

if __name__ == "__main__":
    main()