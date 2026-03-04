import unittest
from powers import square, cube, abs_value, quad, sqrt_approx
class TestPowersEdge(unittest.TestCase):
    def test_square_zero(self): self.assertEqual(square(0), 0)
    def test_sqrt_neg(self): self.assertEqual(sqrt_approx(-5), 0)
    def test_abs_neg(self): self.assertEqual(abs_value(-10), 10)
    def test_quad_one(self): self.assertEqual(quad(1), 1)