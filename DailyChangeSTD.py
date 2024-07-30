import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import tempfile


def save_plot_to_file(plt):
    temp_plot_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    plt.savefig(temp_plot_file.name)
    plt.close()
    return temp_plot_file.name

def calculate_expected_prices(latest_close, expected_change_pct):
    expected_price_positive = latest_close * (1 + expected_change_pct / 100)
    expected_price_negative = latest_close * (1 - expected_change_pct / 100)
    return expected_price_positive, expected_price_negative


ticker_symbol = input("Please enter the ticker symbol: ")
data = yf.download(ticker_symbol, period='6mo')

data['Daily Change %'] = ((data['Close'] - data['Open']) / data['Open']) * 100
bins = np.arange(-3, 3.5, 0.5).tolist()
bin_labels = [f"{bins[i]:.1f}% to {bins[i+1]:.1f}%" for i in range(len(bins)-1)]
data['Bin'] = pd.cut(data['Daily Change %'], bins=bins, labels=bin_labels, include_lowest=True)
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
plt.barh(np.arange(len(frequency_table)), frequency_table['Qty'], color='blue', edgecolor='black')
plt.xlabel('Frequency')
plt.ylabel('Daily Change in %')
plt.title(f'Daily Change in Percentage from Open to Close, past 6 months - {ticker_symbol}')
plt.yticks(ticks=np.arange(len(frequency_table)), labels=frequency_table['Bins'])
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
start = '2024-01-25'
end = '2024-07-25'
stock_data = yf.download(ticker_symbol, start, end)
stock_data['Daily_Price_Difference'] = stock_data['Close'] - stock_data['Open']
stock_data['Weekly_Price_Difference'] = stock_data['Close'] - stock_data['Open'].shift(4)
stock_data['Monthly_Price_Difference'] = stock_data['Close'] - stock_data['Open'].shift(19)

daily_std = np.std(stock_data['Daily_Price_Difference'])
weekly_std = np.std(stock_data['Weekly_Price_Difference'].dropna())
monthly_std = np.std(stock_data['Monthly_Price_Difference'].dropna())

current_stock_price = stock_data['Close'].iloc[-1]
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

plt.figure(figsize=(8, 6))
plt.hist(stock_data['Daily_Price_Difference'], bins=30, color='blue', alpha=0.4, label='Daily')
plt.hist(stock_data['Weekly_Price_Difference'].dropna(), bins=30, color='green', alpha=0.4, label='Weekly')
plt.hist(stock_data['Monthly_Price_Difference'].dropna(), bins=30, color='red', alpha=0.4, label='Monthly')
plt.title(f'{ticker_symbol} Price Difference Histograms')
plt.xlabel('Price Difference')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()



import scipy.stats as stats

# Standard deviation analysis
start = '2024-01-25'
end = '2024-07-25'
stock_data = yf.download(ticker_symbol, start, end)
stock_data['Daily_Price_Difference'] = stock_data['Close'] - stock_data['Open']
stock_data['Weekly_Price_Difference'] = stock_data['Close'] - stock_data['Open'].shift(4)
stock_data['Monthly_Price_Difference'] = stock_data['Close'] - stock_data['Open'].shift(19)

daily_std = np.std(stock_data['Daily_Price_Difference'])
weekly_std = np.std(stock_data['Weekly_Price_Difference'].dropna())
monthly_std = np.std(stock_data['Monthly_Price_Difference'].dropna())

# Generate and plot distribution fits for each frequency
for i, (changes, std, label) in enumerate([
    (stock_data['Daily_Price_Difference'], daily_std, 'Daily'),
    (stock_data['Weekly_Price_Difference'].dropna(), weekly_std, 'Weekly'),
    (stock_data['Monthly_Price_Difference'].dropna(), monthly_std, 'Monthly')
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
    plt.axvline(mean_change + std, color='green', linestyle='dashed', linewidth=2, label='+1 STD')
    plt.axvline(mean_change - std, color='green', linestyle='dashed', linewidth=2, label='-1 STD')
    plt.axvline(mean_change + 2 * std, color='yellow', linestyle='dashed', linewidth=2, label='+2 STD')
    plt.axvline(mean_change - 2 * std, color='yellow', linestyle='dashed', linewidth=2, label='-2 STD')
    plt.axvline(mean_change + 3 * std, color='orange', linestyle='dashed', linewidth=2, label='+3 STD')
    plt.axvline(mean_change - 3 * std, color='orange', linestyle='dashed', linewidth=2, label='-3 STD')
    plt.legend()
    plt.show()