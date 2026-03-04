import unittest
from loader import load_config, set_config_value, config_override
from validator import validate_config, validate_required
class TestIntegration(unittest.TestCase):
    def test_load_validate(self): c=load_config("x"); self.assertTrue(validate_config(c))
    def test_override_required(self): c=config_override({"a":1},{"b":2}); self.assertTrue(validate_required(c, ["a","b"]))