import unittest
from prefix import remove_prefix, common_prefix
from substring import find_substring, replace_first
class TestIntegration(unittest.TestCase):
    def test_common_then_find(self): a="hello"; b="help"; cp=common_prefix(a,b); self.assertEqual(find_substring(a, cp), 0)
    def test_replace_prefix(self): s="hello"; self.assertEqual(replace_first(s, "hel", "xxx"), "xxxlo")