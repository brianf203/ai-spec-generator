import unittest
from mergesort import mergesort, mergesort_inplace, mergesort_iterative
class Test(unittest.TestCase):
    def test_sort(self): self.assertEqual(mergesort([3,1,2]), [1,2,3])
    def test_inplace(self): arr=[3,1,2]; mergesort_inplace(arr); self.assertEqual(arr, [1,2,3])
    def test_iter(self): self.assertEqual(mergesort_iterative([3,1,2]), [1,2,3])