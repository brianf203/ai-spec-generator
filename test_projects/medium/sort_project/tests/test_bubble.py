import unittest
from bubble import bubble_sort, bubble_sort_desc, is_sorted, bubble_sort_inplace
class Test(unittest.TestCase):
    def test_sort(self): self.assertEqual(bubble_sort([3,1,2]), [1,2,3])
    def test_desc(self): self.assertEqual(bubble_sort_desc([3,1,2]), [3,2,1])
    def test_sorted(self): self.assertTrue(is_sorted([1,2,3]))
    def test_inplace(self): lst=[3,1,2]; bubble_sort_inplace(lst); self.assertEqual(lst, [1,2,3])