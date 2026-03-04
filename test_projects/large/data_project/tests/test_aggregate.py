import unittest
from aggregate import sum_values, avg_values, min_values, max_values, count_values, group_by, median
class Test(unittest.TestCase):
    def test_sum(self): self.assertEqual(sum_values([1,2,3]), 6)
    def test_avg(self): self.assertEqual(avg_values([2,4]), 3)
    def test_min(self): self.assertEqual(min_values([3,1,2]), 1)
    def test_max(self): self.assertEqual(max_values([1,3,2]), 3)
    def test_count(self): self.assertEqual(count_values([1,2,3]), 3)
    def test_group(self): self.assertEqual(len(group_by([1,2,1], lambda x: x)), 2)
    def test_median(self): self.assertEqual(median([1,2,3]), 2)