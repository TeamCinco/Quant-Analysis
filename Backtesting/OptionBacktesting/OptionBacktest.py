import pandas as pd
from backtesting import Backtest, Strategy

class OptionBuyAndHoldStrategy(Strategy):
    option_type = 'call'  # 'call' or 'put'
    strike_percent = 0  # 0 for at-the-money, positive for out-of-the-money, negative for in-the-money

    def init(self):
        self.options_data = self.I(self.load_options_data)
        self.entry_price = 0
        self.exit_price = 0
        self.option_strike = 0

    def load_options_data(self):
        # Update the path to your options data CSV file
        options_df = pd.read_csv(r"C:\Users\cinco\Desktop\quant practicie\Research\Research Tools\Options\Options Chain\SPY_2024-08-16_options_chain.csv")
        options_df['lastTradeDate'] = pd.to_datetime(options_df['lastTradeDate'], utc=True)
        numeric_columns = ['strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility']
        for col in numeric_columns:
            options_df[col] = pd.to_numeric(options_df[col], errors='coerce')
        return options_df

    def next(self):
        if not self.position:
            current_price = self.data.Close[-1]
            current_date = self.data.index[-1]
            
            options_on_date = self.options_data[self.options_data['lastTradeDate'].dt.date <= current_date.date()]
            options_on_date = options_on_date.sort_values('lastTradeDate').groupby(['strike', 'Type']).last().reset_index()
            options = options_on_date[options_on_date['Type'] == self.option_type.capitalize()]
            target_strike = current_price * (1 + self.strike_percent / 100)
            self.option_strike = options['strike'].iloc[(options['strike'] - target_strike).abs().argsort()[0]]
            option = options[options['strike'] == self.option_strike].iloc[0]
            self.entry_price = option['lastPrice']
            self.buy(size=1, price=self.entry_price)
        
        elif self.data.index[-1] == self.data.index[-1]:  # Checking if it's the last day of the dataset
            current_price = self.data.Close[-1]
            if self.option_type == 'call':
                self.exit_price = max(0, current_price - self.option_strike)
            else:
                self.exit_price = max(0, self.option_strike - current_price)
            self.position.close()

def load_underlying_data(csv_file):
    data = pd.read_csv(csv_file)
    data['lastTradeDate'] = pd.to_datetime(data['lastTradeDate'], utc=True)
    data = data.sort_values('lastTradeDate')
    data = data.set_index('lastTradeDate')
    data['Open'] = data['High'] = data['Low'] = data['Close'] = data['lastPrice']
    return data[['Open', 'High', 'Low', 'Close']].dropna()

def main():
    underlying_data = load_underlying_data(r"C:\Users\cinco\Desktop\quant practicie\Research\Research Tools\Options\Options Chain\SPY_2024-08-16_options_chain.csv")
    bt_call = Backtest(underlying_data, OptionBuyAndHoldStrategy, cash=10000, commission=.002)
    bt_call.run(option_type='call', strike_percent=0)
    print("Call Option Strategy Results:")
    print(bt_call.stats())

    bt_put = Backtest(underlying_data, OptionBuyAndHoldStrategy, cash=10000, commission=.002)
    bt_put.run(option_type='put', strike_percent=0)
    print("\nPut Option Strategy Results:")
    print(bt_put.stats())

if __name__ == "__main__":
    main()
