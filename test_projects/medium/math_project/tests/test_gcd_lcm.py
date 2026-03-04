import unittest
from gcd_lcm import gcd, lcm, gcd_recursive, lcm_many
class Test(unittest.TestCase):
    def test_gcd(self): self.assertEqual(gcd(12,8), 4)
    def test_lcm(self): self.assertEqual(lcm(4,6), 12)
    def test_gcd_rec(self): self.assertEqual(gcd_recursive(12,8), 4)
    def test_lcm_many(self): self.assertEqual(lcm_many([2,3,4]), 12)