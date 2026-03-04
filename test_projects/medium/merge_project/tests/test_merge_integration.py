import unittest
from merge import merge_sorted_lists, merge_with_duplicates
from mergesort import mergesort
class TestIntegration(unittest.TestCase):
    def test_merge_then_sort(self): a,b=[3,1],[2,4]; merged=merge_sorted_lists(sorted(a), sorted(b)); self.assertEqual(merged, [1,2,3,4])
    def test_dup_merge(self): self.assertEqual(merge_with_duplicates([1,1,2],[2,3]), [1,2,3])