import unittest
from validator import validate_config, merge_configs, validate_required, validate_types, sanitize_config
class Test(unittest.TestCase):
    def test_val(self): self.assertTrue(validate_config({}))
    def test_merge(self): self.assertEqual(merge_configs({"a":1},{"b":2}), {"a":1,"b":2})
    def test_required(self): self.assertTrue(validate_required({"a":1}, ["a"]))
    def test_types(self): self.assertTrue(validate_types({"a":1}, {"a":int}))
    def test_sanitize(self): self.assertEqual(sanitize_config({"a":1,"b":2}, ["a"]), {"a":1})