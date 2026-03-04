import unittest
from primes import is_prime, next_prime, prev_prime, nth_prime, prime_factors, count_primes_up_to
class Test(unittest.TestCase):
    def test_prime(self): self.assertTrue(is_prime(17))
    def test_next(self): self.assertEqual(next_prime(10), 11)
    def test_prev(self): self.assertEqual(prev_prime(10), 7)
    def test_nth(self): self.assertEqual(nth_prime(0), 2)
    def test_factors(self): self.assertEqual(prime_factors(12), [2,2,3])
    def test_count(self): self.assertEqual(count_primes_up_to(10), 4)