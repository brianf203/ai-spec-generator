import unittest
from primes import is_prime, next_prime, prev_prime, prime_factors, count_primes_up_to
class TestEdge(unittest.TestCase):
    def test_not_prime(self): self.assertFalse(is_prime(1))
    def test_prime_two(self): self.assertTrue(is_prime(2))
    def test_prev_prime_none(self): self.assertIsNone(prev_prime(2))
    def test_prime_factors_one(self): self.assertEqual(prime_factors(1), [])
    def test_count_zero(self): self.assertEqual(count_primes_up_to(1), 0)