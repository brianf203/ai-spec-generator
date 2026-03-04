import unittest
from type_checks import is_int, is_str, is_list, is_dict, is_number, coerce_int, safe_int
class Test(unittest.TestCase):
    def test_int(self): self.assertTrue(is_int(1))
    def test_str(self): self.assertTrue(is_str("x"))
    def test_list(self): self.assertTrue(is_list([]))
    def test_dict(self): self.assertTrue(is_dict({}))
    def test_number(self): self.assertTrue(is_number(1.5))
    def test_coerce(self): self.assertEqual(coerce_int("42"), 42)
    def test_safe(self): self.assertEqual(safe_int(None), 0)