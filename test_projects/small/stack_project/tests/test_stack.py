import unittest
from stack import stack_push, stack_pop, stack_peek, stack_is_empty, stack_size
class TestStack(unittest.TestCase):
    def test_push_pop(self): s=[]; stack_push(s,1); stack_push(s,2); self.assertEqual(stack_pop(s), 2)
    def test_peek(self): s=[1,2]; self.assertEqual(stack_peek(s), 2)
    def test_empty(self): self.assertTrue(stack_is_empty([]))
    def test_size(self): self.assertEqual(stack_size([1,2,3]), 3)