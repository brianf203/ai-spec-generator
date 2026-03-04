import unittest
from ordering import compare_asc, compare_desc, rank_in_list
class TestOrdering(unittest.TestCase):
    def test_asc(self): self.assertEqual(compare_asc(1, 2), -1)
    def test_desc(self): self.assertEqual(compare_desc(1, 2), 1)
    def test_rank(self): self.assertEqual(rank_in_list([3, 1, 2], 2), 1)