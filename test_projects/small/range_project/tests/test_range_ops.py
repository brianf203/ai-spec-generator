import unittest
from range_ops import range_len, range_contains, range_overlaps
class TestRangeOps(unittest.TestCase):
    def test_len(self): self.assertEqual(range_len(range(0, 5)), 5)
    def test_contains(self): self.assertTrue(range_contains(range(1, 4), 2))
    def test_overlaps(self): self.assertTrue(range_overlaps(range(0, 5), range(3, 8)))