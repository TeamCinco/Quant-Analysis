import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def analyze_market_conditions(data):
    # Calculate moving averages
    sma_short = data['Close'].rolling(window=5).mean()
    sma_long = data['Close'].rolling(window=10).mean()

    # Calculate volatility (standard deviation of returns)
    returns = data['Close'].pct_change()
    mean_returns = returns.mean()
    std_returns = returns.std()
    volatility = std_returns

    # Check for extended periods of high volatility
    threshold = mean_returns - std_returns
    negative_std_days = returns < threshold
    negative_std_periods = negative_std_days.rolling(window=62).sum()  # window size based on approx. trading days in 3 months

    bearish = negative_std_periods[-1] > 62  # Check the last period

    return bearish, volatility, returns, data['Close'].iloc[-1], std_returns

def plot_volatility_histogram(returns):
    # Remove NaN values for plotting
    returns = returns.dropna()

    # Plot histogram of returns standard deviation
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=30, color='blue', alpha=0.7)
    plt.title('Histogram of Daily Returns Volatility')
    plt.xlabel('Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def main():
    ticker = input("Enter the stock ticker: ")
    start = "2024-01-26"
    end = "2024-07-29"
    
    data = fetch_data(ticker, start, end)
    bearish, volatility, returns, last_close_price, std_dev = analyze_market_conditions(data)
    
    # Calculating the typical price movement range
    typical_range_high = last_close_price + (last_close_price * volatility)
    typical_range_low = last_close_price - (last_close_price * volatility)

    print(f"Analysis for {ticker}:")
    print(f"{'Bearish' if bearish else 'Not Bearish'} condition detected based on extended high volatility.")
    print(f"Market volatility: {volatility:.2%}")
    print(f"Typical daily price movement range from ${typical_range_low:.2f} to ${typical_range_high:.2f} based on the last close price of ${last_close_price:.2f}.")

    # Plotting the histogram of volatility
    plot_volatility_histogram(returns)

    # Advice based on conditions
    if bearish:
        print("Recommended Strategy: Bear market strategies (e.g., buying puts, short positions)")
    else:
        print("No extended bearish conditions detected. Consider strategies suitable for more stable or bullish markets.")

if __name__ == "__main__":
    main()
