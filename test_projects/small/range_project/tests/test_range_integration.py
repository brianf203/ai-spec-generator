import unittest
from range_ops import range_overlaps, range_merge
from intervals import range_sum
class TestIntegration(unittest.TestCase):
    def test_merge_sum(self): a=range(0, 3); b=range(2, 5); m=range_merge(a, b); self.assertEqual(range_sum(m), 10)
    def test_overlap_merge(self): self.assertTrue(range_overlaps(range(0, 5), range(3, 8)))