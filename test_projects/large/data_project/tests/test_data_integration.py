import unittest
from transform import map_values, pick
from aggregate import sum_values, group_by
from export import to_csv_row, from_csv_row
class TestIntegration(unittest.TestCase):
    def test_map_sum(self): d=map_values({"a":1,"b":2}, lambda x: x*2); self.assertEqual(sum_values(list(d.values())), 6)
    def test_csv_roundtrip(self): row=to_csv_row({"a":1,"b":2}, ["a","b"]); self.assertEqual(from_csv_row(row, ["a","b"])["a"], "1")