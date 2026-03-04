import unittest
from aggregate import mean_list, product_list, min_list
class TestEdge(unittest.TestCase):
    def test_mean_empty(self): self.assertEqual(mean_list([]), 0)
    def test_product_empty(self): self.assertEqual(product_list([]), 1)
    def test_min_empty(self): self.assertRaises(ValueError, lambda: min_list([]))