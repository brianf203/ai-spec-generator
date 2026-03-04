import unittest
from sort import quicksort_inplace, quicksort_copy, select_kth
class TestIntegration(unittest.TestCase):
    def test_inplace_eq_copy(self): lst=[5,2,8,1]; self.assertEqual(quicksort_inplace(list(lst)), quicksort_copy(lst))
    def test_select_sorted(self): lst=[3,1,4,1,5]; sorted_lst=quicksort_copy(lst); self.assertEqual(select_kth(lst[:], 2), sorted_lst[2])