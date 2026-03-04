import unittest
from operations import tuple_max, tuple_min, tuple_first, tuple_last
class TestErrors(unittest.TestCase):
    def test_first_empty(self): self.assertIsNone(tuple_first(()))
    def test_last_empty(self): self.assertIsNone(tuple_last(()))
    def test_max_empty(self): self.assertRaises(ValueError, lambda: tuple_max(()))
    def test_min_empty(self): self.assertRaises(ValueError, lambda: tuple_min(()))