import unittest
from operations import tuple_sum, tuple_product, tuple_avg, tuple_slice, tuple_repeat
class TestEdge(unittest.TestCase):
    def test_sum_empty(self): self.assertEqual(tuple_sum(()), 0)
    def test_product_empty(self): self.assertEqual(tuple_product(()), 1)
    def test_avg_empty(self): self.assertEqual(tuple_avg(()), 0)
    def test_slice(self): self.assertEqual(tuple_slice((1,2,3,4), 1, 3), (2, 3))
    def test_repeat(self): self.assertEqual(tuple_repeat((1,2), 2), (1,2,1,2))