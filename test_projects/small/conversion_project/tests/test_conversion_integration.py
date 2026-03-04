import unittest
from num_conv import str_to_int, int_to_str
from coll_conv import list_to_tuple, tuple_to_list
from bool_conv import str_to_bool
class TestConvIntegration(unittest.TestCase):
    def test_str_int_str(self): self.assertEqual(int_to_str(str_to_int("42")), "42")
    def test_list_tuple_list(self): self.assertEqual(tuple_to_list(list_to_tuple([1,2])), [1,2])
    def test_str_bool(self): self.assertTrue(str_to_bool("yes"))