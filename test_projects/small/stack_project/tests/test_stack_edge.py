import unittest
from stack import stack_pop, stack_peek, stack_clear, stack_copy
class TestEdge(unittest.TestCase):
    def test_pop_empty(self): self.assertRaises(IndexError, lambda: stack_pop([]))
    def test_peek_empty(self): self.assertRaises(IndexError, lambda: stack_peek([]))
    def test_clear(self): s=[1,2]; stack_clear(s); self.assertEqual(s, [])
    def test_copy(self): s=[1,2]; c=stack_copy(s); c.append(3); self.assertEqual(len(s), 2)