import unittest
from budget.dp_planner import optimize_budget

class TestDPPlanner(unittest.TestCase):
    def test_priority_allocation_adds_up(self):
        """
        Test that the sum of the allocated funds (minimum + extra) across all categories
        equals the total budget (within a small delta tolerance).

        Steps:
          1. Define a total budget (e.g., 1000) and a list of categories. 
             Each category includes:
               - "Category": Name of the expense category.
               - "Minimum Expense": The minimum required amount.
               - "Priority": A value that influences extra fund allocation.
          2. Call optimize_budget() to compute the allocation.
          3. Sum up the "Total Allocated" values from the resulting allocation.
          4. Assert that the total allocated is almost equal to the total budget (allowing a small delta for floating point imprecision).
        """
        total_budget = 1000
        categories = [
            {"Category": "Rent", "Minimum Expense": 500, "Priority": 3},
            {"Category": "Food", "Minimum Expense": 300, "Priority": 2},
            {"Category": "Books", "Minimum Expense": 100, "Priority": 1}
        ]
        allocation = optimize_budget(total_budget, categories)
        total_allocated = sum(cat["Total Allocated"] for cat in allocation)
        # Check that the sum of all allocated funds is within 0.5 of the total budget
        self.assertAlmostEqual(total_allocated, total_budget, delta=0.5)

    def test_allocation_respects_minimums(self):
        """
        Test that each category's allocated funds (minimum + extra) are at least equal to its specified minimum expense.

        Steps:
          1. Define a total budget and a list of categories with "Minimum Expense" and "Priority".
          2. Call optimize_budget() to compute the allocation.
          3. Iterate over each category and its corresponding allocation.
          4. Assert that the "Total Allocated" for each category is greater than or equal to its "Minimum Expense".
        """
        total_budget = 1000
        categories = [
            {"Category": "Rent", "Minimum Expense": 500, "Priority": 3},
            {"Category": "Food", "Minimum Expense": 300, "Priority": 2},
            {"Category": "Books", "Minimum Expense": 100, "Priority": 1}
        ]
        allocation = optimize_budget(total_budget, categories)
        for original, result in zip(categories, allocation):
            self.assertGreaterEqual(result["Total Allocated"], original["Minimum Expense"])

    def test_invalid_budget_raises_error(self):
        """
        Test that optimize_budget() raises a ValueError when the total budget is insufficient to cover
        the sum of the minimum expenses of all categories.

        Steps:
          1. Define a scenario where the sum of the "Minimum Expense" values exceeds the total budget.
          2. Call optimize_budget() with these values.
          3. Assert that a ValueError is raised.
          
        Example:
          - Categories: Rent (800), Food (400) → Sum = 1200.
          - Total budget: 1000.
          - Expected behavior: ValueError is raised.
        """
        categories = [
            {"Category": "Rent", "Minimum Expense": 800},
            {"Category": "Food", "Minimum Expense": 400}
        ]
        with self.assertRaises(ValueError):
            optimize_budget(1000, categories)

if __name__ == "__main__":
    unittest.main()

