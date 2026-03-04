import unittest
from operations import union_sets, intersection_sets
from utils import unique_list, set_from_iterable
class TestSetIntegration(unittest.TestCase):
    def test_union_from_list(self): self.assertEqual(union_sets(set_from_iterable([1,2]), {2,3}), {1,2,3})
    def test_unique_then_intersect(self): a=unique_list([1,2,1]); b=unique_list([2,3]); self.assertEqual(len(set(a)&set(b)), 1)