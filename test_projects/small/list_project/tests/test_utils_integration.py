import unittest
from operations import flatten_list
from utils import chunk_list, sum_list, product_list
from search import index_of
class TestListIntegration(unittest.TestCase):
    def test_chunk_sum(self): self.assertEqual(sum_list(chunk_list([1,2,3,4], 2)[0]), 3)
    def test_flatten_product(self): self.assertEqual(product_list(flatten_list([[2],[3]])), 6)
    def test_index_in_chunk(self): self.assertEqual(index_of(chunk_list([1,2,3], 1)[1], 2), 0)