import unittest
from iterative import fibonacci_iterative, fib_list, fib_binet, is_fibonacci, fib_index
class Test(unittest.TestCase):
    def test_fib(self): self.assertEqual(fibonacci_iterative(10), 55)
    def test_list(self): self.assertEqual(fib_list(6), [0,1,1,2,3,5])
    def test_binet(self): self.assertEqual(fib_binet(10), 55)
    def test_is_fib(self): self.assertTrue(is_fibonacci(8))
    def test_index(self): self.assertEqual(fib_index(8), 6)