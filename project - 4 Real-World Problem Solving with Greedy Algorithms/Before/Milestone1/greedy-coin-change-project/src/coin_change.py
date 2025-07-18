"""
Greedy Coin Change Calculator

You'll implement the basic greedy algorithm to solve the coin change problem.
"""

from utils import load_denominations

def greedy_coin_change(amount, denominations):
    """
    STEP 2:
    Implement the greedy algorithm to calculate the minimum number of coins needed.

    Instructions:
    - Loop through the coin denominations starting from the largest.
    - For each denomination, determine the maximum number of coins that fit into the remaining amount.
    - Keep track of how many coins of each denomination you use.
    - Reduce the remaining amount accordingly.
    - After processing all denominations, check if the remaining amount is zero.
    - If exact change cannot be made, print a warning message.

    Example:
    amount = 67, denominations = [25, 10, 5, 1]
    Expected result: {25: 2, 10: 1, 5: 1, 1: 2}

    Return:
    - Dictionary with denomination as key and number of coins as value.
    """

    # Write your code here


def main():
    """
    STEP 3:
    Test your greedy coin change algorithm.

    Instructions:
    - Use the provided utility function 'load_denominations' to load denominations from "../data/coin_denominations.csv".
    - Create a list of test amounts to verify your algorithm, for example: [67, 99, 3].
    - Loop through the test amounts, call your greedy_coin_change function, and print the result clearly.
    
    Your printed output should clearly show:
    - The amount of change being calculated.
    - Which coins (and how many of each) were used.
    - The total number of coins used.

    Example output:
    Making change for 67 cents:
    Coins used: {25: 2, 10: 1, 5: 1, 1: 2}
    Total coins needed: 6
    """

    # Write your code here


if __name__ == "__main__":
    main()
