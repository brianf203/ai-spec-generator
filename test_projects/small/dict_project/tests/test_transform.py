import unittest
from transform import map_values, filter_by_value, rename_key
class TestTransform(unittest.TestCase):
    def test_map_values(self): self.assertEqual(map_values({"a":1,"b":2}, lambda x: x*2), {"a":2,"b":4})
    def test_filter_by_value(self): self.assertEqual(filter_by_value({"a":1,"b":3}, lambda x: x>2), {"b":3})
    def test_rename_key(self): self.assertEqual(rename_key({"a":1},"a","x"), {"x":1})