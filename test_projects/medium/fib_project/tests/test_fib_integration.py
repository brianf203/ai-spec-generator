import unittest
from recursive import fibonacci_recursive, fib_memo
from iterative import fibonacci_iterative, fib_list
from lucas import lucas
class TestIntegration(unittest.TestCase):
    def test_rec_eq_iter(self): self.assertEqual(fibonacci_recursive(15), fibonacci_iterative(15))
    def test_list_len(self): self.assertEqual(len(fib_list(10)), 10)
    def test_memo_eq_rec(self): self.assertEqual(fib_memo(12), fibonacci_recursive(12))