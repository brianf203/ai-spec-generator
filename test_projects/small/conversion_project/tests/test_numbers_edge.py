import unittest
from num_conv import str_to_int, int_to_str, float_to_int, int_to_float, str_to_float
class TestNumEdge(unittest.TestCase):
    def test_str_int_neg(self): self.assertEqual(str_to_int("-5"), -5)
    def test_float_int_neg(self): self.assertEqual(float_to_int(-2.9), -2)
    def test_str_float_neg(self): self.assertEqual(str_to_float("-1.5"), -1.5)
    def test_int_str_zero(self): self.assertEqual(int_to_str(0), "0")