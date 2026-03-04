import unittest
from factorial import factorial
from gcd_lcm import gcd, lcm
from primes import is_prime
from digits import sum_digits
class TestIntegration(unittest.TestCase):
    def test_fact_gcd(self): self.assertEqual(gcd(factorial(5), factorial(4)), 24)
    def test_prime_sum(self): self.assertTrue(is_prime(7)); self.assertEqual(sum_digits(7), 7)
    def test_lcm_two(self): self.assertEqual(lcm(4, 6), 12)