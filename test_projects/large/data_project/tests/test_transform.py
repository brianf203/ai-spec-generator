import unittest
from transform import map_values, filter_keys, rename_keys, pick, omit, invert_dict
class Test(unittest.TestCase):
    def test_map(self): self.assertEqual(map_values({"a":1,"b":2}, lambda x: x*2), {"a":2,"b":4})
    def test_filter(self): self.assertEqual(filter_keys({"a":1,"b":2}, ["a"]), {"a":1})
    def test_rename(self): self.assertEqual(rename_keys({"a":1},{"a":"x"}), {"x":1})
    def test_pick(self): self.assertEqual(pick({"a":1,"b":2}, ["a"]), {"a":1})
    def test_omit(self): self.assertEqual(omit({"a":1,"b":2}, ["a"]), {"b":2})
    def test_invert(self): self.assertEqual(invert_dict({"a":1}), {1:"a"})