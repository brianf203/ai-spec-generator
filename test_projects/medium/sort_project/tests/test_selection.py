import unittest
from selection import selection_sort, selection_sort_desc, min_index, max_index
class Test(unittest.TestCase):
    def test_sort(self): self.assertEqual(selection_sort([3,1,2]), [1,2,3])
    def test_desc(self): self.assertEqual(selection_sort_desc([3,1,2]), [3,2,1])
    def test_min_idx(self): self.assertEqual(min_index([3,1,2]), 1)
    def test_max_idx(self): self.assertEqual(max_index([3,1,2]), 0)