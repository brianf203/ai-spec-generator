import unittest
from heap import heap_pop, heap_peek
class TestEdge(unittest.TestCase):
    def test_pop_empty(self): self.assertRaises(IndexError, lambda: heap_pop([]))
    def test_peek_empty(self): self.assertRaises(IndexError, lambda: heap_peek([]))