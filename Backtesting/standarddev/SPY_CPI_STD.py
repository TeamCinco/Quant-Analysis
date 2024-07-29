import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest

class ImprovedStdDevVolumeStrategy(Strategy):
    window = 30
    std_threshold = 1
    volume_threshold = 0.7
    stop_loss = 0.025
    trailing_stop = 0.01
    max_holding_days = 1

    def init(self):
        self.close = self.data.Close
        self.volume = self.data.Volume

        # Calculate returns
        self.returns = self.I(self.calculate_returns)

        # Calculate rolling mean and std of returns
        self.returns_mean = self.I(self.calculate_rolling_mean, self.returns, self.window)
        self.returns_std = self.I(self.calculate_rolling_std, self.returns, self.window)

        # Calculate rolling mean of volume
        self.volume_mean = self.I(self.calculate_rolling_mean, self.volume, self.window)

        # Add trend following
        self.sma50 = self.I(self.calculate_rolling_mean, self.close, 50)
        self.sma200 = self.I(self.calculate_rolling_mean, self.close, 200)

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
        if len(self.data) < max(self.window, 200):
            return

        current_return = self.returns[-1]
        current_volume = self.volume[-1]
        current_price = self.close[-1]

        # Trend following condition
        uptrend = self.sma50[-1] > self.sma200[-1]

        # Debug print
        print(f"Date: {self.data.index[-1]}, Price: {current_price:.2f}, Return: {current_return:.4f}, Volume: {current_volume}")
        print(f"SMA50: {self.sma50[-1]:.2f}, SMA200: {self.sma200[-1]:.2f}, Uptrend: {uptrend}")

        if not self.position:
            if (current_return < self.returns_mean[-1] - self.std_threshold * self.returns_std[-1] and
                current_volume < self.volume_threshold * self.volume_mean[-1] and
                uptrend):
                self.buy(size=0.95)  # Use 95% of available cash
                self.entry_price = current_price
                self.days_held = 0
                print(f"BUY signal at {current_price:.2f}")
        else:
            self.days_held += 1

            # Check stop loss
            if current_price < self.entry_price * (1 - self.stop_loss):
                self.position.close()
                print(f"SELL signal (stop loss) at {current_price:.2f}")

            # Check trailing stop
            elif current_price < self.entry_price * (1 - self.trailing_stop):
                self.position.close()
                print(f"SELL signal (trailing stop) at {current_price:.2f}")

            # Check max holding period
            elif self.days_held >= self.max_holding_days:
                self.position.close()
                print(f"SELL signal (max holding period) at {current_price:.2f}")

            # Check exit conditions
            elif (current_return > self.returns_mean[-1] + self.std_threshold * self.returns_std[-1] or
                  current_volume > 1.2 * self.volume_mean[-1] or
                  not uptrend):
                self.position.close()
                print(f"SELL signal (exit conditions) at {current_price:.2f}")

            # Update entry price for trailing stop
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
    start = "2020-01-01"
    end = "2024-07-25"
    data = fetch_data(ticker, start, end)
    bt = Backtest(data, ImprovedStdDevVolumeStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    buy_and_hold_annualized_return = calculate_annualized_return(data)
    print(f"Annualized Buy and Hold Return: {buy_and_hold_annualized_return * 100:.2f}%")
    bt.plot()

if __name__ == "__main__":
    main()
