import unittest
from stack import stack_push, stack_pop, stack_size
class TestSuite(unittest.TestCase):
    def test_multi_push(self): s=[]; stack_push(s,1); stack_push(s,2); stack_push(s,3); self.assertEqual(stack_size(s), 3)
    def test_pop_order(self): s=[1,2,3]; self.assertEqual(stack_pop(s), 3); self.assertEqual(stack_pop(s), 2)