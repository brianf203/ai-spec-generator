import unittest
from substring import find_substring, count_substring, replace_first
class TestSubstring(unittest.TestCase):
    def test_find(self): self.assertEqual(find_substring("hello", "ll"), 2)
    def test_count(self): self.assertEqual(count_substring("aaa", "a"), 3)
    def test_replace(self): self.assertEqual(replace_first("hello", "l", "x"), "hexlo")