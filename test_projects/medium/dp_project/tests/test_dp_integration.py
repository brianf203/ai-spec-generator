import unittest
from fib import fib_dp, fib_dp_space
from coin import coin_change_ways
from lcs import lcs, edit_distance
class TestIntegration(unittest.TestCase):
    def test_fib_equiv(self): self.assertEqual(fib_dp(20), fib_dp_space(20))
    def test_coin_ways_positive(self): self.assertGreater(coin_change_ways([1,2,3], 5), 0)
    def test_lcs_edit(self): self.assertEqual(lcs("abc","abc"), 3); self.assertEqual(edit_distance("abc","abc"), 0)