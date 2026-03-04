import unittest
from utils import heap_from_list, heap_merge
from heap import heap_pop
class TestUtils(unittest.TestCase):
    def test_from_list(self): h=heap_from_list([3,1,2]); self.assertEqual(heap_pop(h), 1)
    def test_merge(self): h=heap_merge([1,3],[2,4]); self.assertEqual(heap_pop(h), 1)