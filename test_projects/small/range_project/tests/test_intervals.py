import unittest
from intervals import to_list, range_sum, range_min_max
class TestIntervals(unittest.TestCase):
    def test_to_list(self): self.assertEqual(to_list(range(1, 4)), [1, 2, 3])
    def test_sum(self): self.assertEqual(range_sum(range(1, 5)), 10)
    def test_min_max(self): self.assertEqual(range_min_max(range(2, 6)), (2, 5))