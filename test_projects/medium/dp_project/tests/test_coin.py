import unittest
from coin import coin_change, coin_change_ways, coin_change_min_coins
class Test(unittest.TestCase):
    def test_change(self): self.assertEqual(coin_change([1,2,5], 11), 3)
    def test_ways(self): self.assertEqual(coin_change_ways([1,2,5], 5), 4)
    def test_min(self): self.assertEqual(coin_change_min_coins([1,2,5], 11), 3)