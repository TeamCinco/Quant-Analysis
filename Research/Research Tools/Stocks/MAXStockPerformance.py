#Shivors edited this one for my use. He helped fix the code so i can input my profit/ loss.



import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import scipy.stats as stats


def save_plot_to_file(plt):
    temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_plot_file.name)
    plt.close()
    return temp_plot_file.name

def calculate_expected_prices(latest_close, expected_change_pct):
    expected_price_positive = latest_close * (1 + expected_change_pct / 100)
    expected_price_negative = latest_close * (1 - expected_change_pct / 100)
    return expected_price_positive, expected_price_negative

def calculate_iron_condor_payoff(stock_price, lower_bound, upper_bound, premium_collected):
    if stock_price < lower_bound or stock_price > upper_bound:
        return -premium_collected  # Loss is the premium collected
    else:
        return premium_collected  # Reward is the premium collected

ticker_symbol = input("Please enter the ticker symbol: ")
data = yf.download(ticker_symbol, period='max')

data['Daily Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
bins = np.arange(-3, 3.5, 0.5).tolist()
data['Bin'] = pd.cut(data['Daily Change %'], bins=bins, include_lowest=True)
frequency_table = pd.DataFrame({
    'Bins': data['Bin'].value_counts(sort=False).index.categories,
    'Qty': data['Bin'].value_counts(sort=False).values
})
frequency_table['Qty%'] = (frequency_table['Qty'] / frequency_table['Qty'].sum()) * 100
frequency_table['Cum%'] = frequency_table['Qty%'].cumsum()
frequency_table.sort_values(by='Bins', inplace=True)
frequency_table['Qty%'] = frequency_table['Qty%'].map('{:.2f}%'.format)
frequency_table['Cum%'] = frequency_table['Cum%'].map('{:.2f}%'.format)

print(frequency_table.to_string(index=False))

plt.figure(figsize=(12, 8))
plt.barh(frequency_table['Bins'].astype(str), frequency_table['Qty'], color='blue', edgecolor='black')
plt.xlabel('Frequency')
plt.ylabel('Daily Change in %')
plt.title(f'Daily Change in Percentage from Open to Close, all available data - {ticker_symbol}')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

stats_data = {
    'Up Days': ((data['Close'] > data['Open']).sum(), f"{data[data['Close'] > data['Open']]['Daily Change %'].max():.2f}%"),
    'Down Days': ((data['Close'] < data['Open']).sum(), f"{data[data['Close'] < data['Open']]['Daily Change %'].min():.2f}%"),
    'Average': (data['Daily Change %'].mean(),),
    'ST DEV': (data['Daily Change %'].std(),),
    'Variance': (data['Daily Change %'].var(),),
    'Max': (data['Daily Change %'].max(),),
    'Min': (data['Daily Change %'].min(),)
}
stats_df = pd.DataFrame(stats_data, index=['Value', 'Percent' if 'Percent' in stats_data else '']).T
stats_df['Value'] = stats_df['Value'].astype(float).map('{:.2f}'.format)
stats_df = stats_df.reset_index().rename(columns={'index': 'Statistic'})

print(stats_df.to_string(index=False, header=False))

# Standard deviation analysis
data = yf.download(ticker_symbol, period='max')
data['Daily_Price_Difference'] = data['Close'] - data['Open']
data['Weekly_Price_Difference'] = data['Close'] - data['Open'].shift(4)
data['Monthly_Price_Difference'] = data['Close'] - data['Open'].shift(19)

daily_std = np.std(data['Daily_Price_Difference'])
weekly_std = np.std(data['Weekly_Price_Difference'].dropna())
monthly_std = np.std(data['Monthly_Price_Difference'].dropna())

current_stock_price = data['Close'].iloc[-1]
prices_data = {
    'Frequency': ['Daily', 'Weekly', 'Monthly'],
    '1st Std Deviation (-)': [current_stock_price - daily_std, current_stock_price - weekly_std, current_stock_price - monthly_std],
    '1st Std Deviation (+)': [current_stock_price + daily_std, current_stock_price + weekly_std, current_stock_price + monthly_std],
    '2nd Std Deviation (-)': [current_stock_price - 2 * daily_std, current_stock_price - 2 * weekly_std, current_stock_price - 2 * monthly_std],
    '2nd Std Deviation (+)': [current_stock_price + 2 * daily_std, current_stock_price + 2 * weekly_std, current_stock_price + 2 * monthly_std],
    '3rd Std Deviation (-)': [current_stock_price - 3 * daily_std, current_stock_price - 3 * weekly_std, current_stock_price - 3 * monthly_std],
    '3rd Std Deviation (+)': [current_stock_price + 3 * daily_std, current_stock_price + 3 * weekly_std, current_stock_price + 3 * monthly_std]
}
prices_table = pd.DataFrame(prices_data)

print("Standard Deviations:")
print(prices_table)

# Calculate the reward*probability + loss*0.32 for iron condor strategy
iron_condor_results = []

for i, (std, label) in enumerate([
    (daily_std, 'Daily'),
    (weekly_std, 'Weekly'),
    (monthly_std, 'Monthly')
]):
    lower_bound = current_stock_price - std
    upper_bound = current_stock_price + std
    probability_within_bounds = stats.norm.cdf(upper_bound, current_stock_price, std) - stats.norm.cdf(lower_bound, current_stock_price, std)
    
    principal = 200
    premium_collected = 160  # Assume a fixed premium collected for simplicity
    
    expected_outcome_hit = (principal + premium_collected) * probability_within_bounds
    expected_outcome_miss = -800 * (1 - probability_within_bounds)
    
    expected_outcome = expected_outcome_hit + expected_outcome_miss
    
    iron_condor_results.append({
        'Frequency': label,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound,
        'Probability Within Bounds': probability_within_bounds,
        'Expected outcome (hit)': expected_outcome_hit,
        'Expected outcome (miss)': expected_outcome_miss,
        'Expected outcome (all)': expected_outcome
    })

iron_condor_df = pd.DataFrame(iron_condor_results)

print("\nIron Condor Strategy Analysis:")
print(iron_condor_df)

plt.figure(figsize=(8, 6))
plt.hist(data['Daily_Price_Difference'], bins=30, color='blue', alpha=0.4, label='Daily', density=True)
plt.hist(data['Weekly_Price_Difference'].dropna(), bins=30, color='green', alpha=0.4, label='Weekly', density=True)
plt.hist(data['Monthly_Price_Difference'].dropna(), bins=30, color='red', alpha=0.4, label='Monthly', density=True)
plt.title(f'{ticker_symbol} Price Difference Histograms')
plt.xlabel('Price Difference')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

# Generate and plot distribution fits for each frequency with prices
for i, (changes, std, label) in enumerate([
    (data['Daily_Price_Difference'], daily_std, 'Daily'),
    (data['Weekly_Price_Difference'].dropna(), weekly_std, 'Weekly'),
    (data['Monthly_Price_Difference'].dropna(), monthly_std, 'Monthly')
]):
    mean_change = changes.mean()
    plt.figure(figsize=(10, 6))
    hist_data = plt.hist(changes, bins=30, color='blue', alpha=0.5, density=True, label=f'{label} Price Difference')
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_change, std)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
    plt.title(f'Normal Distribution Fit for {label} Price Differences of {ticker_symbol}')
    plt.xlabel(f'{label} Price Difference')
    plt.ylabel('Density')
    
    plt.axvline(mean_change, color='red', linestyle='dashed', linewidth=2, label='Mean')
    plt.text(mean_change, plt.ylim()[1]*0.8, f'{current_stock_price:.2f}', horizontalalignment='right', color='red')
    
    plt.axvline(mean_change + std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.text(mean_change + std, plt.ylim()[1]*0.7, f'{current_stock_price + std:.2f}', horizontalalignment='right', color='green')
    
    plt.axvline(mean_change - std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    plt.text(mean_change - std, plt.ylim()[1]*0.7, f'{current_stock_price - std:.2f}', horizontalalignment='right', color='green')
    
    plt.axvline(mean_change + 2 * std, color='yellow', linestyle='dashed', linewidth=2, label='+2 STD')
    plt.text(mean_change + 2 * std, plt.ylim()[1]*0.6, f'{current_stock_price + 2 * std:.2f}', horizontalalignment='right', color='yellow')
    
    plt.axvline(mean_change - 2 * std, color='yellow', linestyle='dashed', linewidth=2, label='-2 STD')
    plt.text(mean_change - 2 * std, plt.ylim()[1]*0.6, f'{current_stock_price - 2 * std:.2f}', horizontalalignment='right', color='yellow')
    
    plt.axvline(mean_change + 3 * std, color='orange', linestyle='dashed', linewidth=2, label='+3 STD')
    plt.text(mean_change + 3 * std, plt.ylim()[1]*0.5, f'{current_stock_price + 3 * std:.2f}', horizontalalignment='right', color='orange')
    
    plt.axvline(mean_change - 3 * std, color='orange', linestyle='dashed', linewidth=2, label='-3 STD')
    plt.text(mean_change - 3 * std, plt.ylim()[1]*0.5, f'{current_stock_price - 3 * std:.2f}', horizontalalignment='right', color='orange')
    
    plt.legend()
    plt.show()