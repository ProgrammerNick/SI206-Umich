import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to download monthly returns and dividend data for a single ticker
def download_ticker_data(ticker, start_date, end_date):
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date, interval="1mo")
    
    if data.empty:
        return None
    
    # Calculate monthly returns
    data['Monthly Return'] = data['Close'].pct_change()
    
    # Filter for dividends
    dividends = data[data['Dividends'] > 0]
    
    return data, dividends

# Function to download data for all tickers
def download_all_data(tickers, start_date, end_date):
    all_returns = []
    all_dividends = []
    
    for ticker in tickers:
        print(f"Downloading data for {ticker}...")
        try:
            data, dividends = download_ticker_data(ticker, start_date, end_date)
            if data is not None:
                data['Ticker'] = ticker
                all_returns.append(data)
            if not dividends.empty:
                dividends['Ticker'] = ticker
                all_dividends.append(dividends)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {e}")
    
    # Combine all data into DataFrames
    all_returns_df = pd.concat(all_returns)
    all_dividends_df = pd.concat(all_dividends)
    
    return all_returns_df, all_dividends_df

# Example usage
if __name__ == "__main__":
    # Define the date range
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')  # Last 5 years
    
    # Load a list of all publicly traded companies in the US
    # You can download this list from NASDAQ, NYSE, or AMEX
    # Example: Load from a CSV file
    tickers_df = pd.read_csv("us_tickers.csv")  # Replace with your file path
    tickers = tickers_df['Symbol'].tolist()
    
    # Download data
    returns, dividends = download_all_data(tickers, start_date, end_date)
    
    # Save data to CSV files
    returns.to_csv("monthly_returns.csv", index=False)
    dividends.to_csv("monthly_dividends.csv", index=False)
    
    print("Data download complete. Saved to monthly_returns.csv and monthly_dividends.csv.")