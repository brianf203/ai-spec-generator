import unittest
from operations import add, subtract, multiply, divide, mod, floor_div, clamp_val, sign
class TestOpsEdge(unittest.TestCase):
    def test_add_zero(self): self.assertEqual(add(0, 0), 0)
    def test_mod_zero(self): self.assertEqual(mod(5, 0), 0)
    def test_floor_div_zero(self): self.assertEqual(floor_div(5, 0), 0)
    def test_clamp_inside(self): self.assertEqual(clamp_val(5, 0, 10), 5)
    def test_clamp_below(self): self.assertEqual(clamp_val(-5, 0, 10), 0)
    def test_sign_zero(self): self.assertEqual(sign(0), 0)
    def test_sign_neg(self): self.assertEqual(sign(-3), -1)