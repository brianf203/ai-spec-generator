import unittest
from recursive import fibonacci_recursive, fib_tail, fib_memo
class Test(unittest.TestCase):
    def test_fib(self): self.assertEqual(fibonacci_recursive(10), 55)
    def test_tail(self): self.assertEqual(fib_tail(10), 55)
    def test_memo(self): self.assertEqual(fib_memo(10), 55)