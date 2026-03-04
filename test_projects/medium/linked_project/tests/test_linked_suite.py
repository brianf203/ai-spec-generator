import unittest
from list_ops import length, from_list
class TestSuite(unittest.TestCase):
    def test_length(self): h=from_list([1,2,3]); self.assertEqual(length(h), 3)
    def test_empty_length(self): self.assertEqual(length(None), 0)