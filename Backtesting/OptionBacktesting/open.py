import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta

def get_ticker_symbol():
    return input("Enter the ticker symbol: ").upper()

def get_expiration_dates(ticker):
    stock = yf.Ticker(ticker)
    return stock.options

def display_expiration_dates(dates):
    for i, date in enumerate(dates):
        print(f"{i + 1}: {date}")

def select_expiration_date(dates):
    index = int(input("Select the expiration date by number: ")) - 1
    return dates[index]

def get_option_chain(ticker, expiration_date):
    stock = yf.Ticker(ticker)
    return stock.option_chain(expiration_date)

def get_analysis_period():
    print("Select the analysis period for standard deviation:")
    print("1: Daily")
    print("2: Weekly")
    print("3: Monthly")
    period_choice = int(input("Enter the number corresponding to your choice: "))
    period_dict = {1: 'Daily', 2: 'Weekly', 3: 'Monthly'}
    return period_dict.get(period_choice, 'Daily')

def calculate_stats_and_std(ticker, period):
    data = yf.download(ticker, period='6mo')

    if period == 'Daily':
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['Price_Difference'] = data['Close'] - data['Open']
    elif period == 'Weekly':
        data = data.resample('W').last()
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['Price_Difference'] = data['Close'] - data['Open'].shift(1)
    elif period == 'Monthly':
        data = data.resample('M').last()
        data['Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        data['Price_Difference'] = data['Close'] - data['Open'].shift(1)
    
    period_std = np.std(data['Price_Difference'].dropna())
    current_stock_price = data['Close'].iloc[-1]
    prices_data = {
        '1st Std Deviation (-)': [current_stock_price - period_std],
        '1st Std Deviation (+)': [current_stock_price + period_std],
        '2nd Std Deviation (-)': [current_stock_price - 2 * period_std],
        '2nd Std Deviation (+)': [current_stock_price + 2 * period_std],
        '3rd Std Deviation (-)': [current_stock_price - 3 * period_std],
        '3rd Std Deviation (+)': [current_stock_price + 3 * period_std]
    }
    prices_table = pd.DataFrame(prices_data)

    return prices_table, data, period_std, current_stock_price

def get_std_level():
    while True:
        try:
            std_level = int(input("Enter the standard deviation level (1, 2, or 3): "))
            if std_level in [1, 2, 3]:
                return std_level
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_expected_price():
    while True:
        try:
            expected_price = float(input("Enter the expected price at expiration: "))
            return expected_price
        except ValueError:
            print("Invalid input. Please enter a number.")

def find_best_iron_condor_strikes(option_chain, std_levels, std_level, expected_price):
    puts = option_chain.puts
    calls = option_chain.calls
    std_label = ['1st', '2nd', '3rd'][std_level - 1]
    
    lower_bound = std_levels[f'{std_label} Std Deviation (-)'][0]
    upper_bound = std_levels[f'{std_label} Std Deviation (+)'][0]
    
    # Find suitable put strikes
    sell_put_options = puts[(puts['strike'] <= lower_bound) & (puts['strike'] >= lower_bound - 10)]
    buy_put_options = puts[(puts['strike'] < lower_bound)]
    if not sell_put_options.empty and not buy_put_options.empty:
        sell_put = sell_put_options.iloc[-1]  # Choose the highest strike in range
        buy_put = buy_put_options.iloc[0]     # Choose the lowest strike below sell put
    else:
        sell_put = buy_put = None

    # Find suitable call strikes
    sell_call_options = calls[(calls['strike'] >= upper_bound) & (calls['strike'] <= upper_bound + 10)]
    buy_call_options = calls[(calls['strike'] > upper_bound)]
    if not sell_call_options.empty and not buy_call_options.empty:
        sell_call = sell_call_options.iloc[0]  # Choose the lowest strike in range
        buy_call = buy_call_options.iloc[0]    # Choose the lowest strike above sell call
    else:
        sell_call = buy_call = None

    if sell_put is not None and buy_put is not None and sell_call is not None and buy_call is not None:
        iron_condor = {
            'sell_put': sell_put,
            'buy_put': buy_put,
            'sell_call': sell_call,
            'buy_call': buy_call
        }
        return iron_condor
    else:
        print("No suitable Iron Condor strategy found for the given parameters.")
        return None

class IronCondorStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        pass

def run_backtest(data, iron_condor):
    class IronCondorStrategy(Strategy):
        def init(self):
            self.iron_condor = iron_condor

        def next(self):
            if not self.position:
                self.buy(size=100, limit=self.iron_condor['sell_put']['strike'])
                self.sell(size=100, limit=self.iron_condor['buy_put']['strike'])
                self.buy(size=100, limit=self.iron_condor['sell_call']['strike'])
                self.sell(size=100, limit=self.iron_condor['buy_call']['strike'])

    bt = Backtest(data, IronCondorStrategy, cash=10000, commission=.002)
    stats = bt.run()
    return stats

def main():
    ticker = get_ticker_symbol()
    expiration_dates = get_expiration_dates(ticker)
    display_expiration_dates(expiration_dates)
    selected_date = select_expiration_date(expiration_dates)
    
    period = get_analysis_period()
    std_level = get_std_level()
    expected_price = get_expected_price()
    
    prices_table, data, period_std, current_stock_price = calculate_stats_and_std(ticker, period)
    
    option_chain = get_option_chain(ticker, selected_date)
    
    iron_condor = find_best_iron_condor_strikes(option_chain, prices_table, std_level, expected_price)
    
    if not iron_condor:
        print("No suitable Iron Condor strategy found.")
        return

    print("\nSelected Iron Condor Strategy:")
    print(f"Sell Put: Strike {iron_condor['sell_put']['strike']}, Premium {iron_condor['sell_put']['lastPrice']}")
    print(f"Buy Put: Strike {iron_condor['buy_put']['strike']}, Premium {iron_condor['buy_put']['lastPrice']}")
    print(f"Sell Call: Strike {iron_condor['sell_call']['strike']}, Premium {iron_condor['sell_call']['lastPrice']}")
    print(f"Buy Call: Strike {iron_condor['buy_call']['strike']}, Premium {iron_condor['buy_call']['lastPrice']}")

    # Prepare data for backtesting
    backtest_data = yf.download(ticker, start=data.index[0], end=data.index[-1] + pd.Timedelta(days=30))

    # Run backtest
    backtest_results = run_backtest(backtest_data, iron_condor)

    print("\nBacktest Results:")
    print(f"Total Return: {backtest_results['Return [%]']:.2f}%")
    print(f"Sharpe Ratio: {backtest_results['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate: {backtest_results['Win Rate [%]']:.2f}%")

    # Plot the backtest results
    backtest_results.plot()
    plt.show()

if __name__ == "__main__":
    main()
