import unittest
from request import parse_query_string, parse_json_body, get_bearer_token, validate_method
class TestEdge(unittest.TestCase):
    def test_parse_empty(self): self.assertEqual(parse_query_string(""), {})
    def test_json_empty(self): self.assertEqual(parse_json_body(""), {})
    def test_bearer(self): self.assertEqual(get_bearer_token({"Authorization":"Bearer tok"}), "tok")
    def test_method_invalid(self): self.assertFalse(validate_method("INVALID"))