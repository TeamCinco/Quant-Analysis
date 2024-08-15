import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats

# Fetch S&P 500 data
def get_sp500_data(start_date, end_date):
    sp500 = yf.download('spy', start=start_date, end=end_date)['Adj Close']
    return sp500

# Fetch FRED macro data
def get_fred_data(series_id, start_date, end_date):
    data = pdr.get_data_fred(series_id, start=start_date, end=end_date)
    return data

# Linear Regression between S&P 500 and Macro Indicator
def perform_regression(sp500, macro_data):
    model = LinearRegression()
    X = macro_data.values.reshape(-1, 1)  # Reshape data for sklearn
    y = sp500.values
    model.fit(X, y)
    return model

# Calculate daily standard deviation
def calculate_daily_std(ticker_symbol):
    data = yf.download(ticker_symbol, period='max')
    data['Daily_Price_Difference'] = data['Close'] - data['Open']
    daily_std = np.std(data['Daily_Price_Difference'])
    return daily_std

# Monte Carlo Simulation
def monte_carlo_simulation(current_price, projected_daily_return, stdReturn, T, mc_sims=1000):
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    within_1_std = []
    within_2_std = []
    within_3_std = []

    for m in range(mc_sims):
        Z = np.random.normal(size=T)  # Generate random variables for simulation
        dailyReturns = projected_daily_return + stdReturn * Z  # Simulate daily returns
        portfolio_sims[:, m] = np.cumprod(1 + dailyReturns) * current_price  # Calculate cumulative returns
        
        # Ensure values remain within realistic bounds
        portfolio_sims[:, m] = np.maximum(portfolio_sims[:, m], 0)
        
        # Determine if the path stays within standard deviations
        path = portfolio_sims[:, m]
        if all(abs(path - current_price) < 1 * stdReturn):
            within_1_std.append(path)
        elif all(abs(path - current_price) < 2 * stdReturn):
            within_2_std.append(path)
        elif all(abs(path - current_price) < 3 * stdReturn):
            within_3_std.append(path)
    
    return portfolio_sims, within_1_std, within_2_std, within_3_std

# Example main function to run the analysis
def main():
    # User Inputs
    macro_ticker = input("Enter FRED macroeconomic indicator ticker: ")
    start_date = '2000-01-01'
    end_date = '2024-08-14'
    
    # Ask the user for the prediction period
    prediction_period = int(input("Enter the number of days to predict (minimum 5 days): "))
    if prediction_period < 5:
        print("Prediction period is too short. Setting to minimum of 5 days.")
        prediction_period = 5
    
    # Fetch Data
    sp500_data = get_sp500_data(start_date, end_date)
    macro_data = get_fred_data(macro_ticker, start_date, end_date)
    
    # Align data for analysis
    combined_data = pd.concat([sp500_data, macro_data], axis=1).dropna()
    sp500_aligned = combined_data.iloc[:, 0]
    macro_aligned = combined_data.iloc[:, 1]
    
    # Perform Linear Regression
    regression_model = perform_regression(sp500_aligned, macro_aligned)
    print(f"Regression Coefficient: {regression_model.coef_[0]}")
    print(f"Regression Intercept: {regression_model.intercept_}")
    
    # Visualize the regression
    plt.figure(figsize=(10, 6))
    plt.scatter(macro_aligned, sp500_aligned, alpha=0.5)
    plt.plot(macro_aligned, regression_model.predict(macro_aligned.values.reshape(-1, 1)), color='red')
    plt.xlabel(f'{macro_ticker} Data')
    plt.ylabel('S&P 500 Price')
    plt.title(f'Linear Regression of S&P 500 vs {macro_ticker}')
    plt.show()
    
    # Calculate daily standard deviation
    daily_std = calculate_daily_std('SPY')
    
    # Use regression results and daily std for Monte Carlo Simulation
    mean_return = regression_model.coef_[0]
    current_price = sp500_aligned.iloc[-1]
    
    # Constrain mean return and standard deviation to realistic values
    mean_return = max(min(mean_return, 0.001), -0.001)
    daily_std = max(min(daily_std, current_price * 0.02), current_price * 0.005)
    
    # Run Monte Carlo for the user-specified prediction period
    mc_sims_result, within_1_std, within_2_std, within_3_std = monte_carlo_simulation(current_price, mean_return, daily_std, T=prediction_period)
    
    # Visualization of Monte Carlo Simulation with highlighting
    plt.figure(figsize=(12, 8))
    plt.plot(mc_sims_result, color='lightgray', alpha=0.1)
    
    # Highlight paths within each standard deviation range
    plt.plot(within_1_std, color='green', alpha=0.6, label='Within 1 STD')
    plt.plot(within_2_std, color='yellow', alpha=0.6, label='Within 2 STD')
    plt.plot(within_3_std, color='red', alpha=0.6, label='Within 3 STD')
    
    plt.ylabel('S&P 500 Price ($)')
    plt.xlabel('Days')
    plt.title(f'Monte Carlo Simulation of S&P 500 over {prediction_period} days')
    plt.legend()
    
    # Add statistics to the plot
    textstr = f'Current Price: ${current_price:.2f}\n'
    textstr += f'Daily STD: ${daily_std:.2f}\n'
    textstr += f'Mean Return: {mean_return*100:.4f}%'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()