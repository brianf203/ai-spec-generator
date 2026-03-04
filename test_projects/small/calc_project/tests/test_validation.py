import unittest
from validation import validate_divisor, validate_positive, validate_non_neg, validate_in_range, validate_finite
class TestVal(unittest.TestCase):
    def test_ok(self): self.assertTrue(validate_divisor(1))
    def test_zero(self): self.assertRaises(ValueError, lambda: validate_divisor(0))
    def test_positive(self): self.assertTrue(validate_positive(1))
    def test_non_neg(self): self.assertTrue(validate_non_neg(0))
    def test_in_range(self): self.assertTrue(validate_in_range(5, 0, 10))
    def test_finite(self): self.assertTrue(validate_finite(100))