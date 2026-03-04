import unittest
from stack import stack_push, stack_pop, stack_is_empty
from utils import create_stack
class TestIntegration(unittest.TestCase):
    def test_push_pop_cycle(self): s=create_stack(); stack_push(s,1); stack_push(s,2); stack_pop(s); self.assertEqual(stack_pop(s), 1)
    def test_empty_after_pop_all(self): s=[1]; stack_pop(s); self.assertTrue(stack_is_empty(s))