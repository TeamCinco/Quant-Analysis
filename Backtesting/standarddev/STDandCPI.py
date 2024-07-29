import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

class StdDevCPIStrategy(Strategy):
    def init(self):
        self.daily_changes = self.data.Close - self.data.Open

        # Fetch CPI core data and calculate daily returns
        self.cpi_core_data = fetch_fred_data('CPILFESL', self.data.index.min(), self.data.index.max())
        
        # Align CPI core data with trading days
        self.cpi_core_data = self.cpi_core_data.reindex(self.data.index, method='ffill')
        
        self.cpi_core_returns = calculate_daily_returns(self.cpi_core_data)
        
        # Calculate mean and standard deviation of CPI core returns
        self.cpi_core_mean = self.cpi_core_returns.mean().iloc[0] if isinstance(self.cpi_core_returns.mean(), pd.Series) else self.cpi_core_returns.mean()
        self.cpi_core_std = self.cpi_core_returns.std().iloc[0] if isinstance(self.cpi_core_returns.std(), pd.Series) else self.cpi_core_returns.std()

    def next(self):
        if len(self.daily_changes) > 0:
            # Get the latest CPI core return
            latest_date = self.data.index[-1]
            if latest_date in self.cpi_core_returns.index:
                latest_cpi_return = self.cpi_core_returns.loc[latest_date]

                # Check if latest_cpi_return is a Series and extract the scalar value
                if isinstance(latest_cpi_return, pd.Series):
                    latest_cpi_return = latest_cpi_return.iloc[0]  # Get the scalar value

                # Generate buy signal if CPI core return is below -2 std deviations
                if latest_cpi_return < self.cpi_core_mean - 2 * self.cpi_core_std:
                    self.buy()
                else:
                    self.sell()
        else:
            # Handle the case where there is not enough data
            pass

def fetch_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    data.index = pd.to_datetime(data.index)
    return data

def get_available_dates(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")
    return hist.index.min(), hist.index.max()

def calculate_daily_returns(data):
    returns = data.pct_change().dropna()
    return returns

def fetch_fred_data(series_id, start_date, end_date):
    data = web.DataReader(series_id, 'fred', start=start_date, end=end_date)
    return data

def calculate_annualized_return(data):
    total_return = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1
    days = (data.index[-1] - data.index[0]).days
    annualized_return = (1 + total_return) ** (365 / days) - 1
    return annualized_return

def main():
    ticker = input("Enter the ticker symbol: ").upper()
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    start = input(f"Enter the start date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    end = input(f"Enter the end date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, StdDevCPIStrategy, cash=10000, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    
    # Calculate and print the annualized Buy and Hold return
    buy_and_hold_annualized_return = calculate_annualized_return(data)
    print(f"Annualized Buy and Hold Return: {buy_and_hold_annualized_return * 100:.2f}%")
    
    bt.plot()

if __name__ == "__main__":
    main()
