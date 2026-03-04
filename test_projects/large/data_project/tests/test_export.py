import unittest
from export import to_csv_row, from_csv_row, to_json, from_json, escape_csv_val
class Test(unittest.TestCase):
    def test_to(self): self.assertEqual(to_csv_row({"a":1,"b":2}, ["a","b"]), "1,2")
    def test_from(self): self.assertEqual(from_csv_row("1,2", ["a","b"]), {"a":"1","b":"2"})
    def test_json(self): self.assertEqual(from_json(to_json({"a":1})), {"a":1})
    def test_escape(self): self.assertIn("\"", escape_csv_val("a,b"))