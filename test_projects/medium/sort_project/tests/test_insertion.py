import unittest
from insertion import insertion_sort, insertion_sort_desc, insert_sorted, shell_sort
class Test(unittest.TestCase):
    def test_sort(self): self.assertEqual(insertion_sort([3,1,2]), [1,2,3])
    def test_desc(self): self.assertEqual(insertion_sort_desc([3,1,2]), [3,2,1])
    def test_insert(self): self.assertEqual(insert_sorted([1,3], 2), [1,2,3])
    def test_shell(self): self.assertEqual(shell_sort([3,1,2]), [1,2,3])