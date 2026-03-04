import unittest
from validation import validate_divisor, validate_in_range, validate_finite
class TestValErrors(unittest.TestCase):
    def test_divisor_zero_raises(self): self.assertRaises(ValueError, lambda: validate_divisor(0))
    def test_in_range_false(self): self.assertFalse(validate_in_range(15, 0, 10))
    def test_in_range_invalid_lo_hi(self): self.assertFalse(validate_in_range(5, 10, 0))
    def test_finite_large(self): self.assertTrue(validate_finite(1e10))