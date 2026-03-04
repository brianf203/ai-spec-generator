import unittest
from prefix import is_prefix, is_suffix
from substring import count_substring
class TestSuite(unittest.TestCase):
    def test_prefix_suffix(self): self.assertTrue(is_prefix("hello", "hel")); self.assertTrue(is_suffix("hello", "lo"))
    def test_count(self): self.assertEqual(count_substring("aaa", "a"), 3)