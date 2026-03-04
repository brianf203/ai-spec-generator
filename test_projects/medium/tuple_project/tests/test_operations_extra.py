"""Additional tuple operation tests."""
import unittest
from operations import tuple_sum, tuple_avg, tuple_count, tuple_len, tuple_first, tuple_last


class TestTupleOperationsExtra(unittest.TestCase):
    def test_sum_single(self):
        self.assertEqual(tuple_sum((5,)), 5)

    def test_avg_single(self):
        self.assertEqual(tuple_avg((10,)), 10.0)

    def test_count_none(self):
        self.assertEqual(tuple_count((1, 2, 3), 5), 0)

    def test_len_empty(self):
        self.assertEqual(tuple_len(()), 0)

    def test_first_single(self):
        self.assertEqual(tuple_first((42,)), 42)

    def test_last_single(self):
        self.assertEqual(tuple_last((42,)), 42)
