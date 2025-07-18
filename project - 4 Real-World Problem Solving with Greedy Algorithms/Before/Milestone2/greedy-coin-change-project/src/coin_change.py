"""
Greedy Coin Change Calculator with Edge Case Handling

Extend your previous greedy coin change implementation to handle special cases.
"""

from utils import load_denominations, validate_denominations

def greedy_coin_change(amount, denominations):
    """
    STEP 2:
    Extend your greedy algorithm to handle edge cases.

    Instructions:
    - Use your existing implementation from Milestone 1 as the base.
    - After the greedy loop, add logic to verify if exact change was made.
    - If exact change isn't possible, raise a ValueError clearly indicating this issue.

    Example:
    denominations = [25, 10, 5]
    amount = 7 → Cannot make exact change (should raise ValueError)
    """

    # Write your code here (extend your previous Milestone 1 code here)


def main():
    """
    STEP 3:
    Update the main testing function to verify your improvements.

    Instructions:
    - Load and validate denominations.
    - Add test cases including some amounts that cannot be exactly matched by given denominations.
    - Clearly print results or exceptions when exact change isn't possible.

    Suggested Test Cases:
    amounts = [67, 99, 3, 7] (assuming denominations [25, 10, 5, 1])
    amounts = [7, 3] (assuming denominations [25, 10, 5]) — should trigger ValueError
    """

    # Write your code here (expand on your Milestone 1 main function)

if __name__ == "__main__":
    main()
