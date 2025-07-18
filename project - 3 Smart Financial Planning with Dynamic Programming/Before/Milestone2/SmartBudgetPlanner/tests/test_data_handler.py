import unittest
from unittest.mock import patch
from budget.data_handler import get_total_budget, get_categories

class TestDataHandler(unittest.TestCase):
    @patch("builtins.input", side_effect=["3000"])
    def test_get_total_budget_valid(self, mock_input):
        """
        Test that get_total_budget() returns the correct float value when a valid input is provided.
        
        Steps:
          1. Patch the built-in input() function to simulate the user entering "3000".
          2. Call get_total_budget() and assert that the function returns 3000.0.
          
        Expected Behavior:
          If the user enters "3000", the function should return 3000.0 as a float.
        """
        # Write your code here
        pass

    @patch("builtins.input", side_effect=["abc", "-100", "0", "3500"])
    def test_get_total_budget_invalid_then_valid(self, mock_input):
        """
        Test that get_total_budget() eventually returns a valid float value after a series of invalid inputs.
        
        Steps:
          1. Patch input() to simulate a sequence of invalid entries ("abc", "-100", "0")
             followed by a valid entry "3500".
          2. The function should reject the invalid inputs and finally return 3500.0 when given the valid input.
          
        Expected Behavior:
          The function should ignore the invalid values and return 3500.0.
        """
        self.assertEqual(get_total_budget(), 3500.0)

    @patch("builtins.input", side_effect=[
        "Rent", "1000",   # First category: "Rent" with a minimum expense of 1000.
        "Food", "400",    # Second category: "Food" with a minimum expense of 400.
        ""                # Blank input to indicate that no more categories will be added.
    ])
    def test_get_categories_basic(self, mock_input):
        """
        Test that get_categories() correctly processes valid category inputs.
        
        Steps:
          1. Patch input() to simulate the user entering:
              - "Rent" followed by "1000",
              - "Food" followed by "400",
              - then a blank entry to finish.
          2. The function should return a list with two dictionaries representing the categories.
          
        Expected Output:
          [
              {"Category": "Rent", "Minimum Expense": 1000.0},
              {"Category": "Food", "Minimum Expense": 400.0}
          ]
        """
        result = get_categories()
        expected = [
            {"Category": "Rent", "Minimum Expense": 1000.0},
            {"Category": "Food", "Minimum Expense": 400.0}
        ]
        self.assertEqual(result, expected)

    @patch("builtins.input", side_effect=[
        "Utilities",  # Category name "Utilities"
        "abc",        # Invalid expense input (should trigger an error message)
        ""            # Blank input to finish; no valid expense was provided so the category is skipped.
    ])
    def test_get_categories_with_invalid_expense(self, mock_input):
        """
        Test that get_categories() handles an invalid expense input correctly.
        
        Steps:
          1. Patch input() to simulate the user entering "Utilities" as the category name.
          2. Then, the user enters an invalid expense "abc" (which cannot be converted to a float).
          3. Finally, the user provides a blank input to finish, indicating that no valid expense was entered.
          4. The function is expected to skip adding the "Utilities" category.
          
        Expected Output:
          An empty list, since the invalid expense prevents the category from being added.
        """
        result = get_categories()
        expected = []  # Because the invalid expense input should result in no category being added.
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
