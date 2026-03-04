import unittest
from num_conv import str_to_int, int_to_str, float_to_int, int_to_float, str_to_float
class TestNum(unittest.TestCase):
    def test_str_int(self): self.assertEqual(str_to_int("42"), 42)
    def test_int_str(self): self.assertEqual(int_to_str(42), "42")
    def test_float_int(self): self.assertEqual(float_to_int(3.7), 3)
    def test_int_float(self): self.assertEqual(int_to_float(5), 5.0)
    def test_str_float(self): self.assertEqual(str_to_float("3.14"), 3.14)