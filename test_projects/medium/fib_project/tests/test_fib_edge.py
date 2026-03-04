import unittest
from recursive import fibonacci_recursive, fib_tail
from iterative import fibonacci_iterative, is_fibonacci, fib_index
class TestEdge(unittest.TestCase):
    def test_fib_zero(self): self.assertEqual(fibonacci_recursive(0), 0)
    def test_fib_one(self): self.assertEqual(fibonacci_recursive(1), 1)
    def test_tail_zero(self): self.assertEqual(fib_tail(0), 0)
    def test_is_fib_zero(self): self.assertTrue(is_fibonacci(0))
    def test_fib_index_neg(self): self.assertEqual(fib_index(-1), -1)