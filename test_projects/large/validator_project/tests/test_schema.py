import unittest
from schema import validate_schema, get_schema_keys, validate_schema_types, validate_schema_required, schema_merge
class Test(unittest.TestCase):
    def test_val(self): self.assertTrue(validate_schema({"a":1}, ["a"]))
    def test_keys(self): self.assertEqual(get_schema_keys({"a":1}), ["a"])
    def test_types(self): self.assertTrue(validate_schema_types({"a":1}, {"a":int}))
    def test_required(self): self.assertTrue(validate_schema_required({"a":1}, ["a"]))
    def test_merge(self): self.assertEqual(schema_merge({"a":1},{"b":2}), {"a":1,"b":2})