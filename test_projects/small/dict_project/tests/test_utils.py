import unittest
from utils import filter_dict, dict_to_list, get_or_default, keys_list, values_list
class TestUtils(unittest.TestCase):
    def test_filter(self): self.assertEqual(filter_dict({"a":1,"b":2}, ["a"]), {"a":1})
    def test_to_list(self): self.assertEqual(dict_to_list({"a":1}), [("a",1)])
    def test_get_or_default(self): self.assertEqual(get_or_default({"a":1}, "b", 0), 0)
    def test_keys_list(self): self.assertEqual(sorted(keys_list({"a":1,"b":2})), ["a","b"])
    def test_values_list(self): self.assertEqual(sorted(values_list({"a":1,"b":2})), [1,2])