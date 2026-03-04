import unittest
from operations import tuple_sum, tuple_product, tuple_max, tuple_min, tuple_reverse, tuple_avg
from operations import tuple_count, tuple_index, tuple_concat, tuple_sorted, tuple_contains, tuple_len, tuple_first, tuple_last
class Test(unittest.TestCase):
    def test_sum(self): self.assertEqual(tuple_sum((1,2,3)), 6)
    def test_product(self): self.assertEqual(tuple_product((2,3,4)), 24)
    def test_max(self): self.assertEqual(tuple_max((1,3,2)), 3)
    def test_min(self): self.assertEqual(tuple_min((1,3,2)), 1)
    def test_reverse(self): self.assertEqual(tuple_reverse((1,2,3)), (3,2,1))
    def test_avg(self): self.assertEqual(tuple_avg((2,4)), 3.0)
    def test_count(self): self.assertEqual(tuple_count((1,2,1), 1), 2)
    def test_index(self): self.assertEqual(tuple_index((1,2,3), 2), 1)
    def test_concat(self): self.assertEqual(tuple_concat((1,),(2,)), (1,2))
    def test_sorted(self): self.assertEqual(tuple_sorted((3,1,2)), (1,2,3))
    def test_contains(self): self.assertTrue(tuple_contains((1,2), 2))
    def test_len(self): self.assertEqual(tuple_len((1,2,3)), 3)
    def test_first(self): self.assertEqual(tuple_first((1,2)), 1)
    def test_last(self): self.assertEqual(tuple_last((1,2)), 2)