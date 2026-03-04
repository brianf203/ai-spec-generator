import unittest
from merge import merge_dicts
from utils import filter_dict, get_or_default
from transform import map_values
class TestDictIntegration(unittest.TestCase):
    def test_merge_filter(self): self.assertEqual(filter_dict(merge_dicts({"a":1},{"b":2}), ["a"]), {"a":1})
    def test_map_get_default(self): d = map_values({"x":1}, lambda v: v*2); self.assertEqual(get_or_default(d,"x",0), 2)