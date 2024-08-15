import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def get_daily_std(ticker):
    # Fetch data for maximum available time
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    
    # Calculate daily returns
    hist['Daily_Return'] = hist['Close'].pct_change()
    
    # Calculate daily standard deviation
    daily_std = hist['Daily_Return'].std()
    
    # Convert to dollar amount based on the latest closing price
    dollar_std = daily_std * hist['Close'].iloc[-1]
    
    return dollar_std

def screen_stocks(file_path):
    # Read tickers from the file
    with open(file_path, 'r') as file:
        tickers = [line.strip().split()[0] for line in file if line.strip()]
    
    results = []
    errors = []
    
    # Create progress bar
    for ticker in tqdm(tickers, desc="Screening Stocks"):
        try:
            std = get_daily_std(ticker)
            if 3 <= std <= 5:  # -1 STD to +1 STD range of $3
                results.append((ticker, std))
        except Exception as e:
            errors.append((ticker, str(e)))
    
    return results, errors

if __name__ == "__main__":
    file_path = r"C:\Users\cinco\Desktop\quant practicie\Research\Research Tools\Options\ticker.txt.txt"
    screened_stocks, errors = screen_stocks(file_path)
    
    print("\nQuick View of Results:")
    print(f"Total stocks screened: {len(screened_stocks) + len(errors)}")
    print(f"Stocks meeting criteria: {len(screened_stocks)}")
    print(f"Errors encountered: {len(errors)}")
    
    if screened_stocks:
        print("\nStocks meeting the criteria:")
        for ticker, std in screened_stocks:
            print(f"{ticker}: Daily STD = ${std:.2f}")
    else:
        print("\nNo stocks met the criteria.")
    
    if errors:
        print("\nErrors encountered:")
        for ticker, error in errors:
            print(f"{ticker}: {error}")

    print("\nScreening complete. Check the output above for detailed results.")