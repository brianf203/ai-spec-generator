import unittest
from heap import heap_peek, heapify
from utils import heap_from_list
class TestSuite(unittest.TestCase):
    def test_peek_heapify(self): h=heapify([3,1,2]); self.assertEqual(heap_peek(h), 1)
    def test_from_list(self): h=heap_from_list([4,2,6]); self.assertEqual(h[0], 2)