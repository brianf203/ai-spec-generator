import unittest
from aggregate import sum_list, mean_list, product_list
from rounding import round_to_n
class TestIntegration(unittest.TestCase):
    def test_mean_round(self): self.assertEqual(round_to_n(mean_list([1, 2, 3]), 2), 2.0)
    def test_sum_product(self): self.assertEqual(sum_list([1, 2]) + product_list([1, 2]), 5)