import unittest
from prefix import is_prefix, is_suffix, remove_prefix, common_prefix
class TestPrefix(unittest.TestCase):
    def test_prefix(self): self.assertTrue(is_prefix("hello", "hel"))
    def test_suffix(self): self.assertTrue(is_suffix("hello", "lo"))
    def test_remove(self): self.assertEqual(remove_prefix("hello", "hel"), "lo")
    def test_common(self): self.assertEqual(common_prefix("hello", "help"), "hel")