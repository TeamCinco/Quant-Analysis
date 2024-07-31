import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from backtesting import Backtest, Strategy
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
    
    bins = np.arange(-3, 3.5, 0.5).tolist()
    bin_labels = [f"{bins[i]:.1f}% to {bins[i+1]:.1f}%" for i in range(len(bins)-1)]
    data['Bin'] = pd.cut(data['Change %'], bins=bins, labels=bin_labels, include_lowest=True)
    frequency_table = pd.DataFrame({
        'Bins': data['Bin'].value_counts(sort=False).index.categories,
        'Qty': data['Bin'].value_counts(sort=False).values
    })
    frequency_table['Qty%'] = (frequency_table['Qty'] / frequency_table['Qty'].sum()) * 100
    frequency_table['Cum%'] = frequency_table['Qty%'].cumsum()
    frequency_table.sort_values(by='Bins', inplace=True)
    frequency_table['Qty%'] = frequency_table['Qty%'].map('{:.2f}%'.format)
    frequency_table['Cum%'] = frequency_table['Cum%'].map('{:.2f}%'.format)

    period_std = np.std(data['Price_Difference'].dropna())
    current_stock_price = data['Close'].iloc[-1]
    prices_data = {
        'Frequency': [period],
        '1st Std Deviation (-)': [current_stock_price - period_std],
        '1st Std Deviation (+)': [current_stock_price + period_std],
        '2nd Std Deviation (-)': [current_stock_price - 2 * period_std],
        '2nd Std Deviation (+)': [current_stock_price + 2 * period_std],
        '3rd Std Deviation (-)': [current_stock_price - 3 * period_std],
        '3rd Std Deviation (+)': [current_stock_price + 3 * period_std]
    }
    prices_table = pd.DataFrame(prices_data)

    return frequency_table, prices_table, data, period_std, current_stock_price

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

def convert_bin_to_numeric(bin_str):
    try:
        parts = bin_str.replace('%', '').split('to')
        if len(parts) == 2:
            return float(parts[1].strip())
        elif len(parts) == 1:
            return float(parts[0].strip())
        else:
            raise ValueError(f"Unexpected bin format: {bin_str}")
    except ValueError as e:
        print(f"Error converting bin to numeric: {e}")
        return np.nan

def find_best_iron_condor_strikes(option_chain, std_levels, ticker, expiration_date, frequency_table, current_stock_price, std_level, expected_price):
    puts = option_chain.puts
    calls = option_chain.calls
    
    frequency_table['Bins_numeric'] = frequency_table['Bins'].apply(convert_bin_to_numeric)
    frequency_table = frequency_table.dropna(subset=['Bins_numeric'])
    
    std_label = ['1st', '2nd', '3rd'][std_level - 1]
    
    lower_bound = std_levels[f'{std_label} Std Deviation (-)'].values[0]
    upper_bound = std_levels[f'{std_label} Std Deviation (+)'].values[0]
    
    # Find suitable put strikes
    sell_put_options = puts[(puts['strike'] <= lower_bound) & (puts['strike'] >= lower_bound - 10)]
    if not sell_put_options.empty:
        sell_put = sell_put_options.iloc[-1]  # Choose the highest strike in range
        buy_put_options = puts[puts['strike'] < sell_put['strike']]
        buy_put = buy_put_options.iloc[-1] if not buy_put_options.empty else None
    else:
        sell_put = buy_put = None

    # Find suitable call strikes
    sell_call_options = calls[(calls['strike'] >= upper_bound) & (calls['strike'] <= upper_bound + 10)]
    if not sell_call_options.empty:
        sell_call = sell_call_options.iloc[0]  # Choose the lowest strike in range
        buy_call_options = calls[calls['strike'] > sell_call['strike']]
        buy_call = buy_call_options.iloc[0] if not buy_call_options.empty else None
    else:
        sell_call = buy_call = None

    if (sell_put is not None and buy_put is not None and 
        sell_call is not None and buy_call is not None):
        iron_condor = {
            'sell_put': sell_put,
            'buy_put': buy_put,
            'sell_call': sell_call,
            'buy_call': buy_call,
            'std': f'{std_level} Std Dev'
        }
        return iron_condor
    else:
        print(f"No suitable options found for {std_level} standard deviation level.")
        return None

def main():
    ticker = get_ticker_symbol()
    expiration_dates = get_expiration_dates(ticker)
    display_expiration_dates(expiration_dates)
    selected_date = select_expiration_date(expiration_dates)
    
    period = get_analysis_period()
    std_level = get_std_level()
    expected_price = get_expected_price()
    
    frequency_table, prices_table, data, period_std, current_stock_price = calculate_stats_and_std(ticker, period)
    
    option_chain = get_option_chain(ticker, selected_date)
    
    iron_condor = find_best_iron_condor_strikes(option_chain, prices_table, ticker, selected_date, frequency_table, current_stock_price, std_level, expected_price)
    
    if iron_condor is None:
        print("No suitable Iron Condor strategy found for the given parameters.")
        return

    print("\nSelected Iron Condor Strategy:")
    print(f"Sell Put: Strike {iron_condor['sell_put']['strike']}, Premium {iron_condor['sell_put']['lastPrice']}")
    print(f"Buy Put: Strike {iron_condor['buy_put']['strike']}, Premium {iron_condor['buy_put']['lastPrice']}")
    print(f"Sell Call: Strike {iron_condor['sell_call']['strike']}, Premium {iron_condor['sell_call']['lastPrice']}")
    print(f"Buy Call: Strike {iron_condor['buy_call']['strike']}, Premium {iron_condor['buy_call']['lastPrice']}")

    # Calculate max profit and max loss
    max_profit = (iron_condor['sell_put']['lastPrice'] - iron_condor['buy_put']['lastPrice'] + 
                  iron_condor['sell_call']['lastPrice'] - iron_condor['buy_call']['lastPrice']) * 100
    max_loss = (iron_condor['sell_put']['strike'] - iron_condor['buy_put']['strike'] - 
                (iron_condor['sell_put']['lastPrice'] - iron_condor['buy_put']['lastPrice'] + 
                 iron_condor['sell_call']['lastPrice'] - iron_condor['buy_call']['lastPrice'])) * 100

    print(f"\nMax Profit: ${max_profit:.2f}")
    print(f"Max Loss: ${max_loss:.2f}")

    # Calculate potential profit/loss at expected price
    if expected_price <= iron_condor['buy_put']['strike']:
        profit_loss = -max_loss
    elif expected_price >= iron_condor['buy_call']['strike']:
        profit_loss = -max_loss
    elif iron_condor['sell_put']['strike'] <= expected_price <= iron_condor['sell_call']['strike']:
        profit_loss = max_profit
    elif iron_condor['buy_put']['strike'] < expected_price < iron_condor['sell_put']['strike']:
        profit_loss = (expected_price - iron_condor['buy_put']['strike'] + max_profit) * 100
    else:  # iron_condor['sell_call']['strike'] < expected_price < iron_condor['buy_call']['strike']
        profit_loss = (iron_condor['buy_call']['strike'] - expected_price + max_profit) * 100

    print(f"\nPotential Profit/Loss at expected price (${expected_price:.2f}): ${profit_loss:.2f}")

if __name__ == "__main__":
    main()