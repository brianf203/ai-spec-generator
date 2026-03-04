import unittest
from list_ops import from_list, to_list, append, reverse
class TestIntegration(unittest.TestCase):
    def test_append_reverse(self): h=from_list([1]); h=append(h,2); r=reverse(h); self.assertEqual(to_list(r), [2,1])
    def test_roundtrip(self): lst=[1,2,3]; self.assertEqual(to_list(from_list(lst)), lst)