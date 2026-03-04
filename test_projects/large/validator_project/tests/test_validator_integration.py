import unittest
from schema import validate_schema, validate_schema_types, schema_merge
from type_checks import is_int, coerce_int, safe_int
class TestIntegration(unittest.TestCase):
    def test_schema_and_type(self): d={"a":1}; self.assertTrue(validate_schema(d, ["a"]) and validate_schema_types(d, {"a":int}))
    def test_merge_schema(self): s=schema_merge({"a":1},{"b":2}); self.assertEqual(len(s), 2)
    def test_safe_default(self): self.assertEqual(safe_int(None, -1), -1)