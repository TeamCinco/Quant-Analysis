import yfinance as yf
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from datetime import datetime, timedelta, time
from scipy.stats import norm
import pytz
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_risk_free_rate():
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
    symbol = ""
    expiration_date = None
    legs = {}
    exit_condition = 0.0

    def init(self):
        # ... (keep existing initialization)
        self.option_values = {leg: [] for leg in self.legs.keys()}
        self.total_pl = 0

    def next(self):
        try:
            pst = pytz.timezone('America/Los_Angeles')
            current_time = self.data.index[-1].tz_localize(pytz.UTC).astimezone(pst)
            
            if current_time.time() >= time(9, 30) and current_time.time() < time(9, 31):
                if not self.position_opened and current_time.date() == self.entry_date.date():
                    self.open_iron_condor()
                    self.position_opened = True
            
            if self.position_opened and not self.position_closed:
                self.update_option_values()
                self.monitor_position()
            
            if current_time.date() >= self.expiry_date.date() and not self.position_closed:
                self.close_position()
                self.position_closed = True
        except Exception as e:
            logging.error(f"Error in next method: {str(e)}")

    def open_iron_condor(self):
        try:
            current_price = self.data.Close[-1]
            days_to_expiry = (self.expiry_date - self.data.index[-1]).days / 365

            for leg, strike in self.legs.items():
                option_type = 'put' if 'put' in leg else 'call'
                option = self.calc_option(current_price, strike, days_to_expiry, option_type)
                self.option_values[leg].append(option)

            self.premium_received = (self.option_values['short_put'][-1]['price'] + 
                                     self.option_values['short_call'][-1]['price'] - 
                                     self.option_values['long_put'][-1]['price'] - 
                                     self.option_values['long_call'][-1]['price']) * 100
            self.initial_margin = (self.legs['short_put'] - self.legs['long_put']) * 100 - self.premium_received
            logging.info(f"Iron Condor opened. Premium received: {self.premium_received:.2f}")
        except Exception as e:
            logging.error(f"Error in open_iron_condor: {str(e)}")

    def update_option_values(self):
        try:
            current_price = self.data.Close[-1]
            days_to_expiry = (self.expiry_date - self.data.index[-1]).days / 365

            for leg, strike in self.legs.items():
                option_type = 'put' if 'put' in leg else 'call'
                option = self.calc_option(current_price, strike, days_to_expiry, option_type)
                self.option_values[leg].append(option)
        except Exception as e:
            logging.error(f"Error in update_option_values: {str(e)}")

    def monitor_position(self):
        try:
            current_pl = self.calculate_current_pl()
            
            if current_pl <= (1 - self.exit_condition) * self.premium_received:
                self.close_position()
                self.position_closed = True
                logging.info(f"Position closed early. Current P/L: {current_pl:.2f}")
        except Exception as e:
            logging.error(f"Error in monitor_position: {str(e)}")

    def calculate_current_pl(self):
        try:
            current_values = [self.option_values[leg][-1]['price'] for leg in self.legs.keys()]
            current_pl = (current_values[1] + current_values[2] - current_values[0] - current_values[3]) * 100
            return self.premium_received - current_pl
        except Exception as e:
            logging.error(f"Error in calculate_current_pl: {str(e)}")
            return 0

    def close_position(self):
        try:
            final_pl = self.calculate_current_pl()
            self.profits.append(final_pl)
            self.total_pl += final_pl
            logging.info(f"Position closed. Final P/L: {final_pl:.2f}")
        except Exception as e:
            logging.error(f"Error in close_position: {str(e)}")

    def calc_option(self, S, K, T, option_type):
        implied_vol = 0.3  # You might want to use a more sophisticated IV model
        price = black_scholes_option_price(S, K, T, self.risk_free_rate, self.dividend_yield, implied_vol, option_type)
        greeks = calculate_option_greeks(S, K, T, self.risk_free_rate, self.dividend_yield, implied_vol, option_type)
        return {'price': price, **greeks}

def run_backtest(symbol, start_date, end_date, expiration_date, legs, exit_condition):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data available for {symbol} between {start_date} and {end_date}")
        
        data.index = data.index.tz_localize(pytz.UTC)
        
        IronCondorStrategy.symbol = symbol
        IronCondorStrategy.expiration_date = expiration_date
        IronCondorStrategy.legs = legs
        IronCondorStrategy.exit_condition = exit_condition
        
        bt = Backtest(data, IronCondorStrategy, cash=10000, commission=.002)
        results = bt.run()
        logging.info(f"Backtest completed successfully for {symbol}")
        return results
    except Exception as e:
        logging.error(f"Error in run_backtest: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        symbol = input("Enter the ticker symbol: ")
        expiration_date = datetime.strptime(input("Enter the expiration date (YYYY-MM-DD): "), "%Y-%m-%d")
        
        long_put_strike = float(input("Enter the long put strike price: "))
        short_put_strike = float(input("Enter the short put strike price: "))
        short_call_strike = float(input("Enter the short call strike price: "))
        long_call_strike = float(input("Enter the long call strike price: "))

        legs = {
            'long_put': long_put_strike,
            'short_put': short_put_strike,
            'short_call': short_call_strike,
            'long_call': long_call_strike
        }

        exit_condition = float(input("Enter the exit condition as a decimal (e.g., 0.45 for 45% premium): "))

        start_date = "2022-01-01"
        end_date = "2023-12-31"

        results = run_backtest(symbol, start_date, end_date, expiration_date, legs, exit_condition)
        if results is not None:
            print(results)
        else:
            print("Backtest failed. Check the logs for more information.")
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print("An error occurred. Check the logs for more information.")