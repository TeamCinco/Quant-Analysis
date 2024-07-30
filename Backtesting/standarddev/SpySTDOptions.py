import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest

class ShortTermMacroStrategy(Strategy):
    window = 5
    std_threshold = 1
    stop_loss = 0.5
    max_holding_days = 62

    def init(self):
        self.close = self.data.Close

        self.returns = self.I(self.calculate_returns)
        self.returns_mean = self.I(self.calculate_rolling_mean, self.returns, self.window)
        self.returns_std = self.I(self.calculate_rolling_std, self.returns, self.window)

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

        if not self.position:
            if (current_return < self.returns_mean[-1] - self.std_threshold * self.returns_std[-1]):
                self.buy(size=0.98)
                self.entry_price = current_price
                self.days_held = 0
        else:
            self.days_held += 1

            if current_price < self.entry_price * (1 - self.stop_loss):
                self.position.close()
            elif self.days_held >= self.max_holding_days:
                self.position.close()
            elif (current_return > self.returns_mean[-1] + self.std_threshold * self.returns_std[-1]):
                self.position.close()

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

def recommend_iron_condor(options_data, current_price, returns_mean, returns_std, std_threshold):
    strike_prices = options_data['Strike'].unique()
    std_move = returns_std[-1] * std_threshold
    lower_bound = current_price - std_move
    upper_bound = current_price + std_move

    below_strike = max([strike for strike in strike_prices if strike < lower_bound], default=None)
    above_strike = min([strike for strike in strike_prices if strike > upper_bound], default=None)

    if above_strike and below_strike:
        long_put_strike = below_strike
        short_put_strike = (lower_bound + long_put_strike) / 2
        short_call_strike = (upper_bound + above_strike) / 2
        long_call_strike = above_strike

        return {
            "Long Put Strike": long_put_strike,
            "Short Put Strike": short_put_strike,
            "Short Call Strike": short_call_strike,
            "Long Call Strike": long_call_strike
        }

    return None

def main():
    ticker = "SPY"
    start = "2024-01-25"
    end = "2024-07-25"
    data = fetch_data(ticker, start, end)
    bt = Backtest(data, ShortTermMacroStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    buy_and_hold_annualized_return = calculate_annualized_return(data)
    print(f"Annualized Buy and Hold Return: {buy_and_hold_annualized_return * 100:.2f}%")
    bt.plot()

    # Extract indicators from the backtest strategy
    strategy = stats._strategy
    returns_mean = strategy.returns_mean
    returns_std = strategy.returns_std
    current_price = data['Close'].iloc[-1]  # Using the last value of close price as the current price

    # Read options data with hard-coded headers
    headers = ["Contract Name", "Last Trade Date (EDT)", "Strike", "Last Price", "Bid", "Ask", "Change", "% Change", "Volume", "Open Interest", "Implied Volatility"]
    options_data = pd.read_csv(r'C:\Users\cinco\Desktop\quant practicie\SPYoptionsdata.csv', skiprows=1, names=headers)

    iron_condor = recommend_iron_condor(options_data, current_price, returns_mean, returns_std, ShortTermMacroStrategy.std_threshold)
    if iron_condor:
        print("Recommended Iron Condor Strategy:")
        print(iron_condor)
    else:
        print("No suitable iron condor strategy found.")

if __name__ == "__main__":
    main()
