import unittest
from ops import bit_not, is_power_of_two
class TestEdge(unittest.TestCase):
    def test_not(self): self.assertEqual(bit_not(0), -1)
    def test_power_two(self): self.assertTrue(is_power_of_two(8))
    def test_power_two_false(self): self.assertFalse(is_power_of_two(7))