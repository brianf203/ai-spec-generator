"""Integration tests for tuple operations."""
import unittest
from operations import tuple_sum, tuple_product, tuple_concat, tuple_sorted, tuple_reverse


class TestTupleIntegration(unittest.TestCase):
    def test_sum_then_product(self):
        t = (2, 3, 4)
        self.assertEqual(tuple_product((tuple_sum(t),)), 9)

    def test_concat_sorted(self):
        a, b = (3, 1, 2), (6, 4, 5)
        combined = tuple_concat(tuple_sorted(a), tuple_sorted(b))
        self.assertEqual(combined, (1, 2, 3, 4, 5, 6))

    def test_reverse_twice(self):
        t = (1, 2, 3)
        self.assertEqual(tuple_reverse(tuple_reverse(t)), t)
