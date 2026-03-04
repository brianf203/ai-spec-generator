import unittest
from loader import get_nested, config_to_env, env_to_config, config_defaults
class TestEdge(unittest.TestCase):
    def test_nested_missing(self): self.assertIsNone(get_nested({},"a.b"))
    def test_config_to_env(self): self.assertEqual(config_to_env({"x":1})["X"], "1")
    def test_env_to_config(self): self.assertEqual(env_to_config({"PREF_a":"1"},"PREF_"), {"a":"1"})
    def test_defaults(self): self.assertEqual(config_defaults({"a":1},{"b":2}), {"b":2,"a":1})