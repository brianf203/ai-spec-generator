import unittest
from utils import chunk_list, zip_lists, sum_list, product_list, all_positive
class TestUtils(unittest.TestCase):
    def test_chunk(self): self.assertEqual(chunk_list([1,2,3,4], 2), [[1,2],[3,4]])
    def test_zip(self): self.assertEqual(zip_lists([1,2],[3,4]), [(1,3),(2,4)])
    def test_sum(self): self.assertEqual(sum_list([1,2,3]), 6)
    def test_product(self): self.assertEqual(product_list([2,3,4]), 24)
    def test_all_positive(self): self.assertTrue(all_positive([1,2,3]))