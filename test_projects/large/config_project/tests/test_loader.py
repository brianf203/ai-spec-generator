import unittest
from loader import load_config, get_config_value, set_config_value, config_keys, config_override
class Test(unittest.TestCase):
    def test_load(self): c=load_config("x"); self.assertIn("path", c)
    def test_get(self): self.assertEqual(get_config_value({},"x","d"), "d")
    def test_set(self): c={}; set_config_value(c,"a",1); self.assertEqual(c["a"], 1)
    def test_keys(self): self.assertEqual(config_keys({"a":1}), ["a"])
    def test_override(self): self.assertEqual(config_override({"a":1},{"a":2}), {"a":2})