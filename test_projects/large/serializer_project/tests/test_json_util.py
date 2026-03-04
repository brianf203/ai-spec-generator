import unittest
from json_util import to_json, from_json, to_json_pretty, from_json_safe, json_keys, json_contains
class Test(unittest.TestCase):
    def test_roundtrip(self): self.assertEqual(from_json(to_json({"a":1})), {"a":1})
    def test_pretty(self): self.assertIn("\n", to_json_pretty({"a":1}))
    def test_safe(self): self.assertEqual(from_json_safe(""), None)
    def test_keys(self): self.assertEqual(json_keys({"a":1}), ["a"])
    def test_contains(self): self.assertTrue(json_contains({"a":1}, "a"))