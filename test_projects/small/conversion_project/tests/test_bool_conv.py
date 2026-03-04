import unittest
from bool_conv import to_bool, int_to_bool, str_to_bool
class TestBool(unittest.TestCase):
    def test_to_bool(self): self.assertTrue(to_bool(1))
    def test_int_to_bool(self): self.assertTrue(int_to_bool(1))
    def test_str_to_bool(self): self.assertTrue(str_to_bool("true"))