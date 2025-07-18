"""
Greedy Coin Change Calculator with Edge Case Handling

This module implements and optimizes the greedy coin change algorithm,
managing cases where exact change may not be possible.
"""

from src.utils import load_denominations, validate_denominations

def greedy_coin_change(amount, denominations):
    """
    Compute minimum coins required using greedy algorithm.

    Args:
        amount (int): Amount to make change for.
        denominations (list): Sorted denominations list (largest first).

    Returns:
        dict: Dictionary of coins and counts used.

    Raises:
        ValueError: If exact change cannot be achieved.
    """
    coins_used = {}
    remaining_amount = amount

    # Iterate over each coin denomination
    for coin in denominations:
        if remaining_amount >= coin:
            num_coins = remaining_amount // coin
            remaining_amount -= num_coins * coin
            coins_used[coin] = num_coins

    # Check if exact change was achieved
    if remaining_amount != 0:
        raise ValueError(
            f"Exact change for {amount} cents is not possible with denominations provided."
        )

    return coins_used

def main():
    """
    Main function to demonstrate algorithm, including edge case handling.

    Loads coin denominations, validates them, and tests multiple scenarios.
    """
    denominations = load_denominations("../data/coin_denominations.csv")

    # Validate denominations for correctness
    try:
        validate_denominations(denominations)
    except ValueError as e:
        print(f"Denomination validation error: {e}")
        return

    # Test amounts (including edge cases)
    test_amounts = [67, 99, 3, 7]

    for amount in test_amounts:
        print(f"\nAttempting to make change for {amount} cents:")
        try:
            coins = greedy_coin_change(amount, denominations)
            total_coins = sum(coins.values())
            print(f"Coins used: {coins}")
            print(f"Total coins needed: {total_coins}")
        except ValueError as e:
            print(e)

if __name__ == "__main__":
    main()
