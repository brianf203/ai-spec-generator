"""Additional config tests."""
import unittest
from loader import load_config
from validator import validate_config, validate_required


class TestConfigExtra(unittest.TestCase):
    def test_validate_empty(self):
        self.assertTrue(validate_config({}))

    def test_load_config(self):
        cfg = load_config("/tmp/x")
        self.assertIn("path", cfg)

    def test_validate_required_present(self):
        self.assertTrue(validate_required({"a": 1, "b": 2}, ["a"]))
