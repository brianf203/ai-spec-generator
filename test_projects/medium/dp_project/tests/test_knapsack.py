import unittest
from knapsack import knapsack_01, knapsack_unbounded, subset_sum, count_subset_sum
class Test(unittest.TestCase):
    def test_01(self): self.assertEqual(knapsack_01([1,2,3],[6,10,12], 5), 22)
    def test_subset(self): self.assertTrue(subset_sum([1,2,3], 5))
    def test_count(self): self.assertEqual(count_subset_sum([1,2,3], 3), 2)