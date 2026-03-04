import unittest
from prefix import is_prefix, remove_prefix, longest_prefix_match
class TestEdge(unittest.TestCase):
    def test_prefix_longer(self): self.assertFalse(is_prefix("hi", "hello"))
    def test_remove_no_match(self): self.assertEqual(remove_prefix("hello", "xyz"), "hello")
    def test_longest(self): self.assertEqual(longest_prefix_match("hello", ["h","he","hel"]), "hel")