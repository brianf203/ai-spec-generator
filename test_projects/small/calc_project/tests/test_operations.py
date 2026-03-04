import unittest
from operations import add, subtract, multiply, divide, power, mod, floor_div, negate, abs_val
from operations import min_two, max_two, clamp_val, is_even_num, is_odd_num, sign
class TestOps(unittest.TestCase):
    def test_add(self): self.assertEqual(add(2, 3), 5)
    def test_subtract(self): self.assertEqual(subtract(5, 3), 2)
    def test_multiply(self): self.assertEqual(multiply(3, 4), 12)
    def test_divide(self): self.assertEqual(divide(10, 2), 5.0)
    def test_divide_zero(self): self.assertRaises(ValueError, lambda: divide(5, 0))
    def test_power(self): self.assertEqual(power(2, 3), 8)
    def test_mod(self): self.assertEqual(mod(10, 3), 1)
    def test_floor_div(self): self.assertEqual(floor_div(10, 3), 3)
    def test_negate(self): self.assertEqual(negate(5), -5)
    def test_abs_val(self): self.assertEqual(abs_val(-5), 5)
    def test_min_two(self): self.assertEqual(min_two(3, 5), 3)
    def test_max_two(self): self.assertEqual(max_two(3, 5), 5)
    def test_clamp_val(self): self.assertEqual(clamp_val(15, 0, 10), 10)
    def test_is_even_num(self): self.assertTrue(is_even_num(4))
    def test_is_odd_num(self): self.assertTrue(is_odd_num(3))
    def test_sign(self): self.assertEqual(sign(5), 1)