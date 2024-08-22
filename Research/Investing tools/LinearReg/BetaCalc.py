import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def fetch_data(ticker):
    return yf.download(ticker)['Close']

def calculate_beta(ticker_data, spy_data):
    # Align the data on dates and drop missing values
    combined_data = pd.concat([ticker_data, spy_data], axis=1).dropna()
    ticker_data_aligned = combined_data.iloc[:, 0]
    spy_data_aligned = combined_data.iloc[:, 1]
    
    
    # Perform linear regression
    model = LinearRegression()
    model.fit(spy_data_aligned.values.reshape(-1, 1), ticker_data_aligned.values)
    
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(spy_data_aligned.values.reshape(-1, 1), ticker_data_aligned.values)
    
    return slope, intercept, r_squared, ticker_data_aligned, spy_data_aligned

def plot_regression(ticker, ticker_data, spy_data, slope, intercept):
    plt.scatter(spy_data, ticker_data, color='blue', label=f'{ticker} vs SPY')
    plt.plot(spy_data, slope * spy_data + intercept, color='red', label=f'Regression Line')
    plt.xlabel('SPY')
    plt.ylabel(ticker)
    plt.title(f'Regression Analysis: {ticker} vs SPY')
    plt.legend()
    plt.show()

def main():
    tickers = input("Enter tickers separated by commas: ").split(',')
    spy_data = fetch_data('SPY')
    
    for ticker in tickers:
        ticker_data = fetch_data(ticker.strip())
        slope, intercept, r_squared, aligned_ticker_data, aligned_spy_data = calculate_beta(ticker_data, spy_data)
        print(f"Ticker: {ticker}")
        print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}, R-squared: {r_squared:.4f}")
        plot_regression(ticker, aligned_ticker_data, aligned_spy_data, slope, intercept)

if __name__ == "__main__":
    main()
