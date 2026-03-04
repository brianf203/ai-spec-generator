import unittest
from coin import coin_change, coin_change_ways
class TestEdge(unittest.TestCase):
    def test_amount_zero(self): self.assertEqual(coin_change([1,2], 0), 0)
    def test_ways_zero(self): self.assertEqual(coin_change_ways([1,2], 0), 1)
    def test_impossible(self): self.assertEqual(coin_change([2,4], 3), -1)