import unittest
from budget.utils import round_to_cents

class TestUtils(unittest.TestCase):
    def test_rounding_normal(self):
        """
        Test round_to_cents() with a typical floating-point number.
        
        Scenario:
          - Input: 123.456
          - Expected output: 123.46
          
        Explanation:
          This test verifies that round_to_cents correctly rounds a number with more than two 
          decimal places using standard rounding rules. The value 123.456 should round to 123.46.
        """
        self.assertEqual(round_to_cents(123.456), 123.46)

    def test_rounding_down(self):
        """
        Test round_to_cents() with a number that should round down.
        
        Scenario:
          - Input: 100.004
          - Expected output: 100.0
          
        Explanation:
          In this case, the fractional part is small enough that the value rounds down.
          The function should round 100.004 to 100.0 (or 100.00, which is equivalent to 100.0).
        """
        self.assertEqual(round_to_cents(100.004), 100.0)

    def test_rounding_up(self):
        """
        Test round_to_cents() with a number that should round up.
        
        Scenario:
          - Input: 199.995
          - Expected output: 200.0
          
        Explanation:
          This test checks that when the fractional part is close to .995, the function rounds 
          the value up to the next whole number. Thus, 199.995 should round to 200.0.
        """
        self.assertEqual(round_to_cents(199.995), 200.0)

if __name__ == "__main__":
    unittest.main()
