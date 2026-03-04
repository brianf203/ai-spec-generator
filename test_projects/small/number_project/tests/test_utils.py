import unittest
from utils import double, halve, increment, decrement, negate
class TestUtils(unittest.TestCase):
    def test_double(self): self.assertEqual(double(5), 10)
    def test_halve(self): self.assertEqual(halve(10), 5)
    def test_increment(self): self.assertEqual(increment(3), 4)
    def test_decrement(self): self.assertEqual(decrement(3), 2)
    def test_negate(self): self.assertEqual(negate(5), -5)