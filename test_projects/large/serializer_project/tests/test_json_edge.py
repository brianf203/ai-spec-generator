import unittest
from json_util import json_get, json_patch, json_diff, json_serializable
class TestEdge(unittest.TestCase):
    def test_get(self): self.assertEqual(json_get({"a":{"b":1}}, "a.b"), 1)
    def test_patch(self): self.assertEqual(json_patch({"a":1},{"b":2}), {"a":1,"b":2})
    def test_diff(self): self.assertEqual(json_diff({"a":1},{"a":2}), {"a":2})
    def test_serializable(self): self.assertTrue(json_serializable({"x":1}))