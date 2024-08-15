import yfinance as yf
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta, time
from scipy.stats import norm
import pytz

def get_risk_free_rate():
    # Return a fixed risk-free rate for simplicity
    return 0.048  # 4.8% as of August 2024

def get_dividend_yield(ticker):
    try:
        return ticker.info['dividendYield']
    except:
        return 0

def black_scholes_option_price(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return np.nan

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return price

def calculate_option_greeks(S, K, T, r, q, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return {'delta': np.nan, 'gamma': np.nan, 'theta': np.nan, 'vega': np.nan, 'rho': np.nan}

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    if option_type == 'call':
        delta = np.exp(-q * T) * N_d1
        theta = (-((S * sigma * np.exp(-q * T) * n_d1) / (2 * sqrt_T)) 
                 - r * K * np.exp(-r * T) * N_d2
                 + q * S * np.exp(-q * T) * N_d1)
    else:
        delta = -np.exp(-q * T) * norm.cdf(-d1)
        theta = (-((S * sigma * np.exp(-q * T) * n_d1) / (2 * sqrt_T)) 
                 + r * K * np.exp(-r * T) * (1 - N_d2)
                 - q * S * np.exp(-q * T) * (1 - N_d1))

    gamma = (n_d1 * np.exp(-q * T)) / (S * sigma * sqrt_T)
    vega = S * sqrt_T * n_d1 * np.exp(-q * T) / 100
    rho = K * T * np.exp(-r * T) * (N_d2 if option_type == 'call' else (1 - N_d2)) / 100

    theta = theta / 365  # Convert theta to daily value

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

class IronCondorStrategy(Strategy):
    def init(self):
        self.entry_date = self.data.index[0]
        self.expiry_date = self.expiration_date  # User-defined expiration date
        self.position_opened = False
        self.position_closed = False
        
        self.long_put_strike = self.legs['long_put']
        self.short_put_strike = self.legs['short_put']
        self.short_call_strike = self.legs['short_call']
        self.long_call_strike = self.legs['long_call']
        
        self.risk_free_rate = get_risk_free_rate()
        self.dividend_yield = get_dividend_yield(yf.Ticker(self.symbol))  # Use self.symbol

    def next(self):
        # Check if it's 9:30 AM PST
        pst = pytz.timezone('America/Los_Angeles')
        now = datetime.now(pst)
        if now.time() >= time(9, 30) and now.time() < time(9, 31):
            if not self.position_opened and self.data.index[-1] == self.entry_date:
                self.open_iron_condor()
                self.position_opened = True
        
        if self.position_opened and not self.position_closed:
            self.monitor_position()
        
        if self.data.index[-1] >= self.expiry_date and not self.position_closed:
            self.close_position()
            self.position_closed = True

    def open_iron_condor(self):
        current_price = self.data.Close[-1]
        days_to_expiry = (self.expiry_date - self.data.index[-1]).days / 365

        # Calculate option prices and greeks
        long_put = self.calc_option(current_price, self.long_put_strike, days_to_expiry, 'put')
        short_put = self.calc_option(current_price, self.short_put_strike, days_to_expiry, 'put')
        short_call = self.calc_option(current_price, self.short_call_strike, days_to_expiry, 'call')
        long_call = self.calc_option(current_price, self.long_call_strike, days_to_expiry, 'call')

        # Calculate net premium
        net_premium = (short_put['price'] + short_call['price'] - long_put['price'] - long_call['price']) * 100

        self.net_premium = net_premium
        self.initial_margin = (self.short_put_strike - self.long_put_strike) * 100 - net_premium

    def monitor_position(self):
        current_price = self.data.Close[-1]
        days_to_expiry = (self.expiry_date - self.data.index[-1]).days / 365

        # Recalculate option prices
        long_put = self.calc_option(current_price, self.long_put_strike, days_to_expiry, 'put')
        short_put = self.calc_option(current_price, self.short_put_strike, days_to_expiry, 'put')
        short_call = self.calc_option(current_price, self.short_call_strike, days_to_expiry, 'call')
        long_call = self.calc_option(current_price, self.long_call_strike, days_to_expiry, 'call')

        # Calculate current position value
        current_value = (short_put['price'] + short_call['price'] - long_put['price'] - long_call['price']) * 100

        # Check if we should close the position early based on user-defined condition
        if current_value <= (1 - self.exit_condition) * self.net_premium:
            self.close_position()
            self.position_closed = True

    def close_position(self):
        current_price = self.data.Close[-1]
        days_to_expiry = max((self.expiry_date - self.data.index[-1]).days / 365, 1/365)

        # Calculate final option prices
        long_put = self.calc_option(current_price, self.long_put_strike, days_to_expiry, 'put')
        short_put = self.calc_option(current_price, self.short_put_strike, days_to_expiry, 'put')
        short_call = self.calc_option(current_price, self.short_call_strike, days_to_expiry, 'call')
        long_call = self.calc_option(current_price, self.long_call_strike, days_to_expiry, 'call')

        # Calculate final position value
        final_value = (short_put['price'] + short_call['price'] - long_put['price'] - long_call['price']) * 100

        # Calculate and record profit/loss
        profit_loss = self.net_premium - final_value
        self.profits.append(profit_loss)

    def calc_option(self, S, K, T, option_type):
        # Assume a constant implied volatility for simplicity
        implied_vol = 0.3  # You might want to use a more sophisticated method to estimate this

        price = black_scholes_option_price(S, K, T, self.risk_free_rate, self.dividend_yield, implied_vol, option_type)
        greeks = calculate_option_greeks(S, K, T, self.risk_free_rate, self.dividend_yield, implied_vol, option_type)

        return {'price': price, **greeks}

def run_backtest(symbol, start_date, end_date, expiration_date, legs, exit_condition):
    data = yf.download(symbol, start=start_date, end=end_date)
    bt = Backtest(data, IronCondorStrategy, cash=10000, commission=.002)
    bt.strategy.symbol = symbol  # Pass the symbol to the strategy
    bt.strategy.expiration_date = expiration_date
    bt.strategy.legs = legs
    bt.strategy.exit_condition = exit_condition
    results = bt.run()
    return results

if __name__ == "__main__":
    # User inputs
    symbol = input("Enter the ticker symbol: ")
    expiration_date = input("Enter the expiration date (YYYY-MM-DD): ")
    option_type = input("Enter the option type (call/put): ")

    # Getting strikes for the legs of the iron condor
    long_put_strike = float(input("Enter the long put strike price: "))
    short_put_strike = float(input("Enter the short put strike price: "))
    short_call_strike = float(input("Enter the short call strike price: "))
    long_call_strike = float(input("Enter the long call strike price: "))

    # Defining the legs of the iron condor
    legs = {
        'long_put': long_put_strike,
        'short_put': short_put_strike,
        'short_call': short_call_strike,
        'long_call': long_call_strike
    }

    # Asking when to close the trade (default condition is 45% or more premium received)
    exit_condition = float(input("Enter the exit condition as a decimal (e.g., 0.45 for 45% premium): "))

    # Define the start and end dates for the backtest
    start_date = "2022-01-01"
    end_date = "2023-12-31"

    # Run the backtest
    results = run_backtest(symbol, start_date, end_date, expiration_date, legs, exit_condition)
    print(results)
