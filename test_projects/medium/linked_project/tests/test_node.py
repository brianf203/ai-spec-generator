import unittest
from node import create_node, get_value, get_next, set_next
class TestNode(unittest.TestCase):
    def test_create(self): n=create_node(1); self.assertEqual(get_value(n), 1)
    def test_next(self): n=create_node(1, create_node(2)); self.assertEqual(get_value(get_next(n)), 2)
    def test_set_next(self): n=create_node(1); set_next(n, create_node(2)); self.assertIsNotNone(get_next(n))