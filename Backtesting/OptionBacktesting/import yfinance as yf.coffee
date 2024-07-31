import yfinance as yf

def get_spy_options_open_price():
    # Download SPY options data
    spy = yf.Ticker("SPY")

    # Get the expiry dates
    expiries = spy.options
    if not expiries:
        print("No options data available for SPY.")
        return

    # Choose the first expiry date for this example
    expiry_date = expiries[0]
    print(f"Fetching options data for expiry date: {expiry_date}")

    # Get the options chain for the chosen expiry date
    options_chain = spy.option_chain(expiry_date)

    # Combine calls and puts data
    all_options = options_chain.calls.append(options_chain.puts)

    # Display the open prices
    if 'open' in all_options.columns:
        print(all_options[['contractSymbol', 'strike', 'open']])
    else:
        print("Open price data is not available in the options data.")

if __name__ == "__main__":
    get_spy_options_open_price()
