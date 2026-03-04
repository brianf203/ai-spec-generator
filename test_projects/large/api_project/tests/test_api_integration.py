import unittest
from request import parse_query_string, get_query_param
from response import json_response, success_response
from validation import required_fields, validate_json
class TestIntegration(unittest.TestCase):
    def test_parse_then_param(self): q=parse_query_string("a=1"); self.assertEqual(get_query_param(q,"a"), "1")
    def test_json_success(self): d={"x":1}; self.assertTrue(validate_json(d)); r=success_response(d); self.assertEqual(r["data"], d)