import unittest
from powers import square, cube, abs_value, quad, sqrt_approx
class TestPowers(unittest.TestCase):
    def test_square(self): self.assertEqual(square(5), 25)
    def test_cube(self): self.assertEqual(cube(3), 27)
    def test_abs(self): self.assertEqual(abs_value(-5), 5)
    def test_quad(self): self.assertEqual(quad(2), 16)
    def test_sqrt_approx(self): self.assertEqual(sqrt_approx(10), 3)