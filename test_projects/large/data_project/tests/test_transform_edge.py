import unittest
from transform import flatten_dict, nest_dict, deep_merge, filter_values
class TestEdge(unittest.TestCase):
    def test_flatten(self): self.assertEqual(flatten_dict({"a":{"b":1}}), {"a.b":1})
    def test_nest(self): self.assertEqual(nest_dict({"a.b":1}), {"a":{"b":1}})
    def test_deep_merge(self): self.assertEqual(deep_merge({"a":1},{"b":2}), {"a":1,"b":2})
    def test_filter(self): self.assertEqual(filter_values({"a":1,"b":0}, lambda x: x != 0), {"a":1})