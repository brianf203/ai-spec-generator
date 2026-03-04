import unittest
from validation import validate_json, required_fields, validate_types, validate_range, validate_enum
class Test(unittest.TestCase):
    def test_json(self): self.assertTrue(validate_json({}))
    def test_req(self): self.assertTrue(required_fields({"a":1}, ["a"]))
    def test_types(self): self.assertTrue(validate_types({"a":1}, {"a":int}))
    def test_range(self): self.assertTrue(validate_range(5, 0, 10))
    def test_enum(self): self.assertTrue(validate_enum("a", ["a","b"]))