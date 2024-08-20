import yfinance as yf
import pandas as pd
import datetime

def fetch_historical_price(stock_symbol, date):
    stock = yf.Ticker(stock_symbol)
    hist = stock.history(start=date, end=(datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
    return hist['Close'].iloc[0]

def get_user_input():
    stock_symbol = input("Enter the stock symbol: ")
    date = input("Enter the date (YYYY-MM-DD): ")
    strategy = input("Select a strategy (vertical spread or iron condor): ").lower()
    strike_preferences = input("Enter preferred strike prices (comma separated): ").split(',')
    strike_preferences = [float(sp.strip()) for sp in strike_preferences]
    return stock_symbol, date, strategy, strike_preferences

def read_options_data(file_path):
    return pd.read_csv(file_path)

def filter_options_data(df, stock_symbol):
    return df[df['act_symbol'] == stock_symbol]

def find_vertical_spreads(options_data, strike_preferences):
    vertical_spreads = []
    for exp_date in options_data['expiration'].unique():
        for strike in strike_preferences:
            call_options = options_data[(options_data['expiration'] == exp_date) & (options_data['call_put'] == 'call')]
            put_options = options_data[(options_data['expiration'] == exp_date) & (options_data['call_put'] == 'put')]
            
            call_leg1 = call_options[call_options['strike'] == strike]
            call_leg2 = call_options[call_options['strike'] == strike + 1]
            put_leg1 = put_options[put_options['strike'] == strike]
            put_leg2 = put_options[put_options['strike'] == strike + 1]

            if not call_leg1.empty and not call_leg2.empty:
                vertical_spreads.append({
                    'type': 'vertical spread (call)',
                    'expiration': exp_date,
                    'strike1': strike,
                    'strike2': strike + 1,
                    'call_put': 'call',
                    'bid1': call_leg1.iloc[0]['bid'],
                    'ask1': call_leg1.iloc[0]['ask'],
                    'bid2': call_leg2.iloc[0]['bid'],
                    'ask2': call_leg2.iloc[0]['ask']
                })

            if not put_leg1.empty and not put_leg2.empty:
                vertical_spreads.append({
                    'type': 'vertical spread (put)',
                    'expiration': exp_date,
                    'strike1': strike,
                    'strike2': strike + 1,
                    'call_put': 'put',
                    'bid1': put_leg1.iloc[0]['bid'],
                    'ask1': put_leg1.iloc[0]['ask'],
                    'bid2': put_leg2.iloc[0]['bid'],
                    'ask2': put_leg2.iloc[0]['ask']
                })
    return vertical_spreads

def find_iron_condors(options_data, strike_preferences, stock_price):
    iron_condors = []
    for exp_date in options_data['expiration'].unique():
        call_options = options_data[(options_data['expiration'] == exp_date) & (options_data['call_put'] == 'call')]
        put_options = options_data[(options_data['expiration'] == exp_date) & (options_data['call_put'] == 'put')]
        
        for strike in strike_preferences:
            call_leg1 = call_options[(call_options['strike'] == stock_price + strike)]
            call_leg2 = call_options[(call_options['strike'] == stock_price + strike + 1)]
            put_leg1 = put_options[(put_options['strike'] == stock_price - strike)]
            put_leg2 = put_options[(put_options['strike'] == stock_price - strike - 1)]

            if not call_leg1.empty and not call_leg2.empty and not put_leg1.empty and not put_leg2.empty:
                iron_condors.append({
                    'type': 'iron condor',
                    'expiration': exp_date,
                    'call_strike1': stock_price + strike,
                    'call_strike2': stock_price + strike + 1,
                    'put_strike1': stock_price - strike,
                    'put_strike2': stock_price - strike - 1,
                    'call_put': 'both',
                    'call_bid1': call_leg1.iloc[0]['bid'],
                    'call_ask1': call_leg1.iloc[0]['ask'],
                    'call_bid2': call_leg2.iloc[0]['bid'],
                    'call_ask2': call_leg2.iloc[0]['ask'],
                    'put_bid1': put_leg1.iloc[0]['bid'],
                    'put_ask1': put_leg1.iloc[0]['ask'],
                    'put_bid2': put_leg2.iloc[0]['bid'],
                    'put_ask2': put_leg2.iloc[0]['ask']
                })
    return iron_condors

def display_strategies(stock_price, strategies):
    table = pd.DataFrame(strategies)
    print(f"Stock price on specified date: {stock_price}")
    print(table)

def main():
    stock_symbol, date, strategy, strike_preferences = get_user_input()
    stock_price = fetch_historical_price(stock_symbol, date)
    options_data = read_options_data('/Users/jazzhashzzz/Documents/options_master_1722147453750.csv.csv')
    filtered_options = filter_options_data(options_data, stock_symbol)

    if strategy == 'vertical spread':
        strategies = find_vertical_spreads(filtered_options, strike_preferences)
    elif strategy == 'iron condor':
        strategies = find_iron_condors(filtered_options, strike_preferences, stock_price)
    else:
        print("Invalid strategy selected.")
        return

    display_strategies(stock_price, strategies)

if __name__ == "__main__":
    main()
