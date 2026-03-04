"""Additional validator tests."""
import unittest
from schema import validate_schema
from type_checks import is_str, is_number


class TestValidatorExtra(unittest.TestCase):
    def test_validate_schema_empty(self):
        self.assertTrue(validate_schema({}, {}))

    def test_is_str(self):
        self.assertTrue(is_str("hello"))

    def test_is_number(self):
        self.assertTrue(is_number(42))
