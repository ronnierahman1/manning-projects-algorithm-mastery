import yfinance as yf
import matplotlib.pyplot as plt

# Milestone 4: Evaluating and Visualizing Trading Performance
# -------------------------------------------------------------
# This module provides two functions:
#
# 1. evaluate_strategy(data, close_col)
#    - This function calculates the percentage returns based on the specified close column
#      in the given DataFrame.
#    - It then computes the cumulative returns (by taking the cumulative product of (1 + returns))
#      and plots the resulting performance over time.
#
# 2. plot_trading_signals(data, close_col, decisions)
#    - This function plots the stock's closing price over time.
#    - It then overlays buy/sell signals based on the provided decisions list.
#      Each decision is a tuple containing (action, date, price).
#    - Buy signals are marked with a green upward-pointing marker and Sell signals with a red downward-pointing marker.
#
# Note: Both functions display plots using matplotlib's plt.show(), which will open a window
#       with the plot when called.
#
# Example usage for evaluate_strategy:
#   evaluate_strategy(data, 'Close')
#
# Example usage for plot_trading_signals:
#   plot_trading_signals(data, 'Close', decisions)

def evaluate_strategy(data, close_col):
    print("Evaluating strategy performance using CSV data...", flush=True)
    # Calculate daily percentage returns
    data['Returns'] = data[close_col].pct_change()
    # Calculate cumulative returns over time
    cumulative_returns = (1 + data['Returns']).cumprod()
    # Plot the cumulative returns
    cumulative_returns.plot(title="Cumulative Returns from Historical Data")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.grid(True)
    print("Displaying cumulative returns plot. Close the plot window to continue...", flush=True)
    plt.show()


def plot_trading_signals(data, close_col, decisions):
    # Create a new figure and axis for the plot
    fig, ax = plt.subplots()
    # Plot the closing price over time
    ax.plot(data.index, data[close_col], label='Closing Price', color='blue')
    
    # Loop over each trading decision to add markers and annotations
    for decision in decisions:
        action, date, price = decision
        if action == 'Buy':
            ax.plot(date, price, marker='^', color='green', markersize=10)
            ax.annotate('Buy', (date, price), textcoords="offset points", xytext=(0,10), ha='center', color='green')
        elif action == 'Sell':
            ax.plot(date, price, marker='v', color='red', markersize=10)
            ax.annotate('Sell', (date, price), textcoords="offset points", xytext=(0,-15), ha='center', color='red')
    
    # Set plot title, labels, and legend
    ax.set_title("Stock Price with Buy/Sell Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    print("Displaying trading signals plot. Close the plot window to continue...", flush=True)
    plt.show()


