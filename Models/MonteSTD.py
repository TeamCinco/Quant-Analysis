import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats

# Fetch stock data
def get_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)[['Open', 'Close']]
    return stock_data

# Fetch FRED macro data
def get_fred_data(series_id, start_date, end_date):
    data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
    return data

# Normalize and Log Transform Data
def normalize_and_log(data):
    data = data.dropna()  # Remove any NaN values
    log_data = np.log(data)   # Log transform the data
    normalized_data = (log_data - log_data.min()) / (log_data.max() - log_data.min())  # Normalize data
    return normalized_data, log_data

# De-normalize and De-log data
def denormalize_and_delog(normalized_data, log_data):
    min_val = log_data.min()
    max_val = log_data.max()
    denormalized_data = normalized_data * (max_val - min_val) + min_val  # De-normalize
    original_data = np.exp(denormalized_data)  # De-log (inverse of log is exp)
    return original_data

# Linear Regression between Stock and Macro Indicator
def perform_regression(stock_data, macro_data):
    model = LinearRegression()
    X = macro_data.values.reshape(-1, 1)  # Reshape data for sklearn
    y = stock_data.values
    model.fit(X, y)
    return model

# Monte Carlo Simulation
def monte_carlo_simulation(current_price, projected_daily_return, stdReturn, T, mc_sims=1000000):
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    for m in range(0, mc_sims):
        Z = np.random.normal(size=T)  # Generate random variables for simulation
        dailyReturns = projected_daily_return + stdReturn * Z  # Simulate daily returns
        portfolio_sims[:, m] = np.cumprod(1 + dailyReturns) * current_price  # Calculate cumulative returns
    
    return portfolio_sims

# Updated Analyze Paths Within STD Ranges for Given Timeframe
def analyze_paths_within_std(simulations, current_price, daily_std, weekly_std, monthly_std):
    T, mc_sims = simulations.shape
    std_ranges = [1, 2, 3]
    timeframes = ['Daily', 'Weekly', 'Monthly']
    std_values = [daily_std, weekly_std, monthly_std]
    path_analysis = []
    
    for sim in range(mc_sims):
        path = simulations[:, sim]
        labels = []
        for timeframe, std_value in zip(timeframes, std_values):
            for std in std_ranges:
                lower_bound = current_price - std * std_value
                upper_bound = current_price + std * std_value
                if np.all((path >= lower_bound) & (path <= upper_bound)):
                    labels.append(f"{timeframe} +/- {std} STD")
        
        if labels:
            path_analysis.append((sim, labels))
        else:
            path_analysis.append((sim, ["Outside of all STD ranges"]))
    
    return path_analysis

# Calculate Standard Deviations (from Iron Condor script)
def calculate_std_deviations(data):
    data['Daily_Price_Difference'] = data['Close'] - data['Open']
    data['Weekly_Price_Difference'] = data['Close'] - data['Open'].shift(4)
    data['Monthly_Price_Difference'] = data['Close'] - data['Open'].shift(19)

    daily_std = np.std(data['Daily_Price_Difference'])
    weekly_std = np.std(data['Weekly_Price_Difference'].dropna())
    monthly_std = np.std(data['Monthly_Price_Difference'].dropna())

    return daily_std, weekly_std, monthly_std

# Plot the Standard Deviation Distribution with Stock Prices
def plot_std_distribution(stock_data, std_value, label, stock_ticker):
    # Calculate mean price
    mean_price = stock_data['Close'].iloc[-1]
    
    # Calculate the actual stock price levels for each standard deviation
    std_prices = {
        f"+{i} STD": mean_price + i * std_value for i in range(1, 4)
    }
    std_prices.update({
        f"-{i} STD": mean_price - i * std_value for i in range(1, 4)
    })
    
    # Plot histogram and normal distribution fit
    plt.figure(figsize=(12, 6))
    price_differences = stock_data['Close'].diff().dropna()
    hist_data = plt.hist(price_differences, bins=50, color='blue', alpha=0.5, density=True, label=f'{label} Price Difference')
    
    # Fit normal distribution
    mu, sigma = stats.norm.fit(price_differences)
    x = np.linspace(price_differences.min(), price_differences.max(), 100)
    p = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'k', linewidth=2, label='Normal Distribution Fit')
    
    plt.title(f'Normal Distribution Fit for {label} Price Differences of {stock_ticker}')
    plt.xlabel(f'{label} Price Difference ($)')
    plt.ylabel('Density')
    
    # Add standard deviation lines and labels with stock price figures
    for i, color in zip(range(1, 4), ['green', 'yellow', 'orange']):
        std_price_plus = mean_price + i * std_value
        std_price_minus = mean_price - i * std_value
        plt.axvline(i * std_value, color=color, linestyle='dashed', linewidth=2, label=f'+{i} STD ({std_price_plus:.2f})')
        plt.axvline(-i * std_value, color=color, linestyle='dashed', linewidth=2, label=f'-{i} STD ({std_price_minus:.2f})')
        plt.text(i * std_value, plt.ylim()[1] * (0.9 - 0.1 * i), f'{std_price_plus:.2f}', horizontalalignment='right', color=color)
        plt.text(-i * std_value, plt.ylim()[1] * (0.9 - 0.1 * i), f'{std_price_minus:.2f}', horizontalalignment='left', color=color)
    
    # Add mean line and label
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2, label=f'Mean ({mean_price:.2f})')
    plt.text(0, plt.ylim()[1] * 0.95, f'{mean_price:.2f}', horizontalalignment='center', color='red')
    
    # Set x-axis limits to a reasonable range (e.g., +/- 3 STD)
    plt.xlim(-3 * std_value, 3 * std_value)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main function to run the analysis and save data to CSV
def main():
    # User Inputs
    stock_ticker = input("Enter the stock ticker symbol: ")
    macro_ticker = input("Enter FRED macroeconomic indicator ticker: ")
    start_date = '2003-01-01'
    end_date = '2024-08-17'
    
    # Ask the user for the prediction period
    prediction_period = int(input("Enter the number of days to predict (minimum 5 days): "))
    if prediction_period < 5:
        print("Prediction period is too short. Setting to minimum of 5 days.")
        prediction_period = 5
    
    # Fetch Data
    stock_data = get_stock_data(stock_ticker, start_date, end_date)
    macro_data = get_fred_data(macro_ticker, start_date, end_date)
    
    # Align data for analysis
    combined_data = pd.concat([stock_data['Close'], macro_data], axis=1).dropna()
    stock_aligned = combined_data.iloc[:, 0]
    macro_aligned = combined_data.iloc[:, 1]
    
    # Normalize and log transform the data
    stock_normalized, stock_log = normalize_and_log(stock_aligned)
    macro_normalized, macro_log = normalize_and_log(macro_aligned)
    
    # Perform Linear Regression
    regression_model = perform_regression(stock_normalized, macro_normalized)
    print(f"Regression Coefficient: {regression_model.coef_[0]}")
    print(f"Regression Intercept: {regression_model.intercept_}")
    
    # De-normalize and de-log the data for Monte Carlo simulation
    stock_original = denormalize_and_delog(stock_normalized, stock_log)
    macro_original = denormalize_and_delog(macro_normalized, macro_log)
    
    # Calculate Standard Deviations using the correct logic
    daily_std, weekly_std, monthly_std = calculate_std_deviations(stock_data)
    
    current_price = stock_original.iloc[-1]
    
    # Monte Carlo Simulation
    mean_return = regression_model.coef_[0]
    std_return = daily_std / current_price  # Convert to a percentage for simulation
    
    # Ensure mean return and standard deviation are within reasonable limits
    mean_return = max(min(mean_return, 0.001), -0.001)
    std_return = max(min(std_return, 0.02), 0.005)
    
    # Run Monte Carlo for the user-specified prediction period
    mc_sims_result = monte_carlo_simulation(current_price, mean_return, std_return, T=prediction_period)
    
    # Analyze paths for daily, weekly, and monthly STD ranges and save data to CSV
    all_paths = pd.DataFrame(mc_sims_result, columns=[f"Path {i+1}" for i in range(mc_sims_result.shape[1])])
    all_paths_info = {}
    
    # Analyze and label paths
    path_analysis = analyze_paths_within_std(mc_sims_result, current_price, daily_std, weekly_std, monthly_std)
    for sim, labels in path_analysis:
        all_paths_info[f"Path {sim+1}"] = "; ".join(labels)
    
    # Add path labels to the DataFrame and save to CSV
    all_paths.columns = [f"{col} ({all_paths_info[col]})" for col in all_paths.columns]
    all_paths.to_csv(f"{stock_ticker}_MC_Paths.csv", index=False)
    
    print(f"Monte Carlo paths and their STD analysis have been saved to {stock_ticker}_MC_Paths.csv")
    
    # Visualization of Monte Carlo Simulation
    plt.figure(figsize=(10, 5))
    plt.plot(mc_sims_result)
    plt.ylabel(f'{stock_ticker} Price ($)')
    plt.xlabel('Days')
    plt.title(f'MC simulation of {stock_ticker} over {prediction_period} days')
    plt.show()
    
    # Show Daily, Weekly, and Monthly STD Charts with Stock Prices
    plot_std_distribution(stock_data, daily_std, 'Daily', stock_ticker)
    plot_std_distribution(stock_data.resample('W').last(), weekly_std, 'Weekly', stock_ticker)
    plot_std_distribution(stock_data.resample('M').last(), monthly_std, 'Monthly', stock_ticker)

if __name__ == "__main__":
    main()