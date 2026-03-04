import unittest
from coll_conv import list_to_tuple, tuple_to_list, bytes_to_str, str_to_bytes, set_to_list
class TestColl(unittest.TestCase):
    def test_list_tuple(self): self.assertEqual(list_to_tuple([1,2]), (1,2))
    def test_tuple_list(self): self.assertEqual(tuple_to_list((1,2)), [1,2])
    def test_bytes_str(self): self.assertEqual(bytes_to_str(b"hi"), "hi")
    def test_str_bytes(self): self.assertEqual(str_to_bytes("hi"), b"hi")
    def test_set_list(self): self.assertEqual(sorted(set_to_list({1,2})), [1,2])