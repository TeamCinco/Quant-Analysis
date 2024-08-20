import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from sklearn.linear_model import LinearRegression

class SmaCross(Strategy):
    def init(self):
        price = pd.Series(self.data.Close, index=self.data.index)
        self.ma1 = self.I(SMA, price, 10)  # Short-term moving average
        self.ma2 = self.I(SMA, price, 20)  # Long-term moving average

        # Calculate the linear regression between S&P 500 and CPI core
        self.cpi_core_data = fetch_fred_data('CPILFESL', self.data.index.min(), self.data.index.max())
        self.sp500_returns = calculate_daily_returns(price)
        self.cpi_core_returns = calculate_daily_returns(self.cpi_core_data)
        
        combined_data = pd.concat([self.sp500_returns, self.cpi_core_returns], axis=1).dropna()
        self.sp500_returns_aligned = combined_data.iloc[:, 0]
        self.cpi_core_returns_aligned = combined_data.iloc[:, 1]
        
        self.regression_model = LinearRegression()
        self.regression_model.fit(self.sp500_returns_aligned.values.reshape(-1, 1), self.cpi_core_returns_aligned.values.reshape(-1, 1))
        
        self.coefficient = self.regression_model.coef_[0][0]
        self.intercept = self.regression_model.intercept_[0]
        self.r2_score = self.regression_model.score(self.sp500_returns_aligned.values.reshape(-1, 1), self.cpi_core_returns_aligned.values.reshape(-1, 1))
        
    def next(self):
        if crossover(self.ma1, self.ma2):
            # Adjust position size based on the linear regression findings
            size = round(max(1, 1 + self.coefficient * self.r2_score))
            self.buy(size=size)  # Buy signal with adjusted size
        elif crossover(self.ma2, self.ma1):
            self.sell()  # Sell signal

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
    # Calculate the total return from start to end
    total_return = data['Close'].iloc[-1] / data['Close'].iloc[0] - 1
    
    # Calculate the number of days in the period
    days = (data.index[-1] - data.index[0]).days
    
    # Annualize the return
    annualized_return = (1 + total_return) ** (365 / days) - 1
    return annualized_return

def main():
    ticker = input("Enter the ticker symbol: ").upper()
    
    start_date, end_date = get_available_dates(ticker)
    print(f"Available data range for {ticker}: {start_date.date()} to {end_date.date()}")
    
    start = input(f"Enter the start date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    end = input(f"Enter the end date (YYYY-MM-DD) between {start_date.date()} and {end_date.date()}: ")
    
    data = fetch_data(ticker, start, end)
    
    bt = Backtest(data, SmaCross, commission=.002, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    
    # Calculate and print the annualized Buy and Hold return
    buy_and_hold_annualized_return = calculate_annualized_return(data)
    print(f"Annualized Buy and Hold Return: {buy_and_hold_annualized_return * 100:.2f}%")
    
    bt.plot()

if __name__ == "__main__":
    main()
