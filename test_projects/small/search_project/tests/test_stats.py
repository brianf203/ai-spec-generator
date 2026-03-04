import unittest
from stats import find_min, find_max, count_occurrences, sum_list, average
class TestStats(unittest.TestCase):
    def test_min(self): self.assertEqual(find_min([3,1,2]), 1)
    def test_max(self): self.assertEqual(find_max([1,3,2]), 3)
    def test_count(self): self.assertEqual(count_occurrences([1,2,1], 1), 2)
    def test_sum(self): self.assertEqual(sum_list([1,2,3]), 6)
    def test_average(self): self.assertEqual(average([2,4]), 3.0)