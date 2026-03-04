import unittest
from linear import linear_search, find_all_indices
from stats import find_min, count_occurrences, average
from filters import filter_positive
class TestSearchIntegration(unittest.TestCase):
    def test_min_after_filter(self): self.assertEqual(find_min(filter_positive([3,1,-1,2])), 1)
    def test_count_indices(self): lst=[1,2,1]; self.assertEqual(len(find_all_indices(lst,1)), count_occurrences(lst,1))
    def test_avg_filtered(self): self.assertEqual(average(filter_positive([2,4])), 3.0)