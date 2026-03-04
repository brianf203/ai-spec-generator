import unittest
from filters import filter_positive, filter_even, filter_unique, take_while, drop_while
class TestFilters(unittest.TestCase):
    def test_positive(self): self.assertEqual(filter_positive([1,-1,2]), [1,2])
    def test_even(self): self.assertEqual(filter_even([1,2,3,4]), [2,4])
    def test_unique(self): self.assertEqual(filter_unique([1,2,1]), [1,2])
    def test_take_while(self): self.assertEqual(take_while([1,2,3], lambda x: x<3), [1,2])
    def test_drop_while(self): self.assertEqual(drop_while([1,2,3], lambda x: x<2), [2,3])