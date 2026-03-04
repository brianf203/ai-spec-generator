import unittest
from schema import schema_pick, schema_omit, schema_diff, schema_keys_intersection
class TestEdge(unittest.TestCase):
    def test_pick(self): self.assertEqual(schema_pick({"a":1,"b":2}, ["a"]), {"a":1})
    def test_omit(self): self.assertEqual(schema_omit({"a":1,"b":2}, ["a"]), {"b":2})
    def test_diff(self): self.assertEqual(schema_diff({"a":1},{"a":2}), {"a":1})
    def test_intersection(self): self.assertEqual(schema_keys_intersection({"a":1},{"a":2}), ["a"])