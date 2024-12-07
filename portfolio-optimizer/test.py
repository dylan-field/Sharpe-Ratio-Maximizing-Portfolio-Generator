import yfinance as yf
import pandas as pd
from datetime import datetime

# List of stock tickers from the image
tickers = [
    "LLY", "PWR", "MCK", "GIS", "ENPH", "MRNA", "CMG", "TSLA", "NVDA", "PGR", "SPY"
]

# Define the date range
start_date = "2023-12-15"  # Start date mentioned
end_date = datetime.now().strftime("%Y-%m-%d")  # Today's date

# Fetch data for adjusted close prices
print(f"Fetching adjusted close prices from {start_date} to {end_date}...")
data = yf.download(tickers, start=start_date, end=end_date, interval="1d", progress=False)

# Extract adjusted close prices
adj_close_data = data["Adj Close"]

# Check if data is fetched
if adj_close_data.empty:
    print("No data fetched. Please check the ticker symbols or date range.")
else:
    # Remove timezone information from index
    adj_close_data.index = adj_close_data.index.tz_localize(None)
    
    # Save to Excel
    output_file = r"C:\Users\dkaif\OneDrive\Desktop\Sharpe_Ratio_Testing\Adjusted_Close_Prices.xlsx"
    adj_close_data.to_excel(output_file)
    print(f"Adjusted close prices saved to {output_file}")
