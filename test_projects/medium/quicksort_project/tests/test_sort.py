import unittest
from sort import quicksort_inplace, quicksort_copy, quicksort_hoare, select_kth
class Test(unittest.TestCase):
    def test_sort(self): lst=[3,1,2]; quicksort_inplace(lst); self.assertEqual(lst, [1,2,3])
    def test_copy(self): self.assertEqual(quicksort_copy([3,1,2]), [1,2,3])
    def test_hoare(self): lst=[3,1,2]; quicksort_hoare(lst); self.assertEqual(lst, [1,2,3])
    def test_select(self): self.assertEqual(select_kth([3,1,4,2], 2), 3)