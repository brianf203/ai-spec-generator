import unittest
from aggregate import sum_list, product_list, mean_list, min_list, max_list
class TestAggregate(unittest.TestCase):
    def test_sum(self): self.assertEqual(sum_list([1, 2, 3]), 6)
    def test_product(self): self.assertEqual(product_list([2, 3, 4]), 24)
    def test_mean(self): self.assertEqual(mean_list([2, 4]), 3)
    def test_min(self): self.assertEqual(min_list([3, 1, 2]), 1)
    def test_max(self): self.assertEqual(max_list([1, 3, 2]), 3)