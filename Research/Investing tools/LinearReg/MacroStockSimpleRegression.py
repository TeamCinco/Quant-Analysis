import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr

def fetch_stock_data(ticker):
    data = yf.download(ticker)
    return data['Adj Close']

def fetch_fred_data(fred_ticker):
    fred_data = pdr.get_data_fred(fred_ticker)
    return fred_data['Value']

def calculate_beta(stock_data, fred_data):
    # Align the data to the common date range
    common_dates = stock_data.index.intersection(fred_data.index)
    stock_data = stock_data.loc[common_dates]
    fred_data = fred_data.loc[common_dates]

    # Reshape data for linear regression
    X = fred_data.values.reshape(-1, 1)
    y = stock_data.values
    
    # Perform linear regression
    model = LinearRegression().fit(X, y)
    beta = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(X, y)
    
    return beta, intercept, r_squared

def plot_beta(ticker, stock_data, fred_data, beta, intercept):
    plt.figure(figsize=(10, 6))
    plt.scatter(fred_data, stock_data, color='blue', label=f'{ticker} vs FRED Variable')
    plt.plot(fred_data, beta * fred_data + intercept, color='red', label='Linear Regression')
    plt.xlabel('FRED Variable')
    plt.ylabel(f'{ticker}')
    plt.title(f'Beta Calculation: {ticker} vs FRED Variable')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    stock_ticker = input("Enter the stock ticker: ").upper()
    fred_ticker = input("Enter the FRED ticker: ").upper()

    # Fetch stock and FRED data
    stock_data = fetch_stock_data(stock_ticker)
    fred_data = fetch_fred_data(fred_ticker)
    
    if not stock_data.empty and not fred_data.empty:
        beta, intercept, r_squared = calculate_beta(stock_data, fred_data)
        print(f"Stock Ticker: {stock_ticker}")
        print(f"FRED Ticker: {fred_ticker}")
        print(f"Beta: {beta:.4f}, Intercept: {intercept:.4f}, R-squared: {r_squared:.4f}")
        plot_beta(stock_ticker, stock_data, fred_data, beta, intercept)
    else:
        print(f"Data not available for {stock_ticker} or {fred_ticker}.")

if __name__ == "__main__":
    main()
