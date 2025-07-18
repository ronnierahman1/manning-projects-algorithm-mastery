# Note: Ensure you have the yfinance library installed in your Python environment.
# You can install it using:
# pip install -r requirements.txt
#
# This script fetches historical and real-time stock data for Apple Inc. (AAPL)
# and saves them to CSV files in the 'data' directory.
#
# Historical data is fetched for the period from 2023-01-01 to 2023-12-31.
# Real-time data is fetched for the past 5 days with 5-minute intervals.

# Import necessary libraries
import yfinance as yf
import pandas as pd

# Fetch historical data for AAPL for the specified date range.
historical_data = yf.download('AAPL', start='2023-01-01', end='2023-12-31')
# Save the historical data to a CSV file in the 'data' directory.
historical_data.to_csv('data/historical_stock_data.csv')

# Fetch recent real-time data for AAPL.
# 'period' set to '5d' means data for the past 5 days.
# 'interval' set to '5m' fetches data at 5-minute intervals.
real_time_data = yf.download('AAPL', period='5d', interval='5m')
# Save the real-time data to a CSV file in the 'data' directory.
real_time_data.to_csv('data/real_time_stock_data.csv')

# Inform the user that the CSV files have been populated.
print("CSV files populated successfully!")
print("Historical data and real-time data have been saved to CSV files.")
