"""
Greedy Coin Change Calculator

This module implements the basic greedy algorithm to solve the coin change problem.
"""

from utils import load_denominations

def greedy_coin_change(amount, denominations):
    """
    Calculate the minimum number of coins needed to make change for the given amount
    using a greedy algorithm.

    Args:
        amount (int): The amount of money (in cents) for which change is required.
        denominations (list): List of coin denominations sorted in descending order.

    Returns:
        dict: A dictionary with denomination as key and number of coins as value.

    Note:
        If exact change is not possible with provided denominations,
        the algorithm returns the closest possible change and issues a warning.
    """
    # Dictionary to store coins used for change
    coins_used = {}

    # Iterate over coin denominations starting from the largest
    for coin in denominations:
        if amount >= coin:
            # Find the maximum number of current coins to use
            num_coins = amount // coin
            amount -= num_coins * coin
            coins_used[coin] = num_coins

    # Check if exact change was possible
    if amount != 0:
        print("Warning: Cannot make exact change with the given denominations.")

    return coins_used

def main():
    """
    Main function to test the greedy coin change algorithm.

    Loads coin denominations from CSV file and tests the algorithm with predefined amounts.
    """
    # Load denominations from CSV file
    denominations = load_denominations("../data/coin_denominations.csv")

    # Test cases for verification
    test_amounts = [67, 99, 3]

    # Loop over each test case and display results
    for amount in test_amounts:
        print(f"\nMaking change for {amount} cents:")
        result = greedy_coin_change(amount, denominations)
        total_coins = sum(result.values())
        print(f"Coins used: {result}")
        print(f"Total coins needed: {total_coins}")

if __name__ == "__main__":
    main()
