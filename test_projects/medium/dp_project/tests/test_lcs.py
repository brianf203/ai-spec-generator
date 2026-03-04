import unittest
from lcs import lcs, lcs_length, edit_distance, lps
class Test(unittest.TestCase):
    def test_lcs(self): self.assertEqual(lcs("abcde","ace"), 3)
    def test_edit(self): self.assertEqual(edit_distance("horse","ros"), 3)
    def test_lps(self): self.assertEqual(lps("bbbab"), 4)