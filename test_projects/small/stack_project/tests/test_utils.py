import unittest
from utils import create_stack, stack_from_list, stack_reverse, stack_contains
class TestUtils(unittest.TestCase):
    def test_create(self): self.assertEqual(create_stack(), [])
    def test_from_list(self): self.assertEqual(stack_from_list([1,2]), [1,2])
    def test_reverse(self): self.assertEqual(stack_reverse([1,2,3]), [3,2,1])
    def test_contains(self): self.assertTrue(stack_contains([1,2], 2))