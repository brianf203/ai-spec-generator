import unittest
from operations import add, subtract, multiply, divide, clamp_val, abs_val
from validation import validate_divisor
class TestOpsIntegration(unittest.TestCase):
    def test_chain_ops(self): self.assertEqual(add(multiply(2, 3), 1), 7)
    def test_clamp_then_abs(self): self.assertEqual(abs_val(clamp_val(-10, 0, 5)), 0)
    def test_validate_before_divide(self):
        validate_divisor(2)
        self.assertEqual(divide(10, 2), 5.0)