import unittest
from bubble import bubble_sort, is_sorted
from selection import selection_sort
from insertion import insertion_sort
class TestIntegration(unittest.TestCase):
    def test_all_same(self): lst=[3,1,2]; self.assertTrue(is_sorted(bubble_sort(lst)))
    def test_selection_equals_insertion(self): lst=[5,2,8,1]; self.assertEqual(selection_sort(lst), insertion_sort(lst))