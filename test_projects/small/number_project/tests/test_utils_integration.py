import unittest
from parity import is_even
from powers import square
from utils import double, halve
class TestNumIntegration(unittest.TestCase):
    def test_double_square(self): self.assertEqual(square(double(3)), 36)
    def test_halve_even(self): self.assertTrue(is_even(halve(8)))