import unittest
from budget.tiered_planner import smart_allocation

class TestSmartAllocation(unittest.TestCase):
    def test_allocation_when_emergency_not_met(self):
        """
        Test the allocation when the emergency fund target is NOT met.
        
        Scenario:
          - Total monthly income: $3000
          - Essential expenses: $2000
          - Remaining funds: $3000 - $2000 = $1000
          - Emergency fund target: $10000
          - Current emergency fund balance: $8000 (target not met)
        
        Expected Behavior:
          - Allocate 70% of the remaining funds to "Emergency Fund": 0.70 * $1000 = $700.
          - Allocate the remaining 30% to "Miscellaneous": 0.30 * $1000 = $300.
        """
        result = smart_allocation(
            total_income=3000,
            essential_total=2000,
            emergency_target=10000,
            current_emergency_balance=8000
        )
        # Write your code here:
        # e.g., assert that result["Emergency Fund"] is approximately 700.0 and result["Miscellaneous"] is approximately 300.0.
        pass

    def test_allocation_when_emergency_met(self):
        """
        Test the allocation when the emergency fund target is met.
        
        Scenario:
          - Total monthly income: $3000
          - Essential expenses: $2000
          - Remaining funds: $3000 - $2000 = $1000
          - Emergency fund target: $10000
          - Current emergency fund balance: $10000 (target met)
        
        Expected Behavior:
          - Allocate 90% of the remaining funds to "Savings/Investments": 0.90 * $1000 = $900.
          - Allocate the remaining 10% to "Miscellaneous": 0.10 * $1000 = $100.
        """
        result = smart_allocation(
            total_income=3000,
            essential_total=2000,
            emergency_target=10000,
            current_emergency_balance=10000
        )
        # Write your code here:
        # e.g., assert that result["Savings/Investments"] is approximately 900.0 and result["Miscellaneous"] is approximately 100.0.
        pass

    def test_allocation_when_expenses_exceed_income(self):
        """
        Test that a ValueError is raised when the essential expenses exceed the total income.
        
        Scenario:
          - Total monthly income: $2000
          - Essential expenses: $2500
          
        Expected Behavior:
          - Since the income is insufficient to cover the minimum required expenses,
            smart_allocation() should raise a ValueError.
        """
        # Write your code here:
        # e.g., use a context manager (with self.assertRaises(ValueError):) to verify that smart_allocation() raises an error.
        pass

if __name__ == "__main__":
    unittest.main()
