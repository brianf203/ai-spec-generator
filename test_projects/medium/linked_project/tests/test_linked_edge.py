import unittest
from list_ops import length, to_list, from_list
class TestEdge(unittest.TestCase):
    def test_empty(self): self.assertEqual(length(None), 0)
    def test_single(self): h=from_list([1]); self.assertEqual(to_list(h), [1])