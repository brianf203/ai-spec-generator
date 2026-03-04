import unittest
from recursive import binary_search_recursive, bsearch_first_ge, bsearch_last_le
class Test(unittest.TestCase):
    def test_bs(self): self.assertEqual(binary_search_recursive([1,2,3,4], 3), 2)
    def test_first_ge(self): self.assertEqual(bsearch_first_ge([1,2,2,3], 2), 1)
    def test_last_le(self): self.assertEqual(bsearch_last_le([1,2,2,3], 2), 2)