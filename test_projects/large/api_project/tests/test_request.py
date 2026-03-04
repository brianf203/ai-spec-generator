import unittest
from request import parse_query_string, get_header, get_query_param, validate_method, get_user_agent
class Test(unittest.TestCase):
    def test_parse(self): self.assertEqual(parse_query_string("a=1&b=2"), {"a":"1","b":"2"})
    def test_header(self): self.assertEqual(get_header({"X":"y"},"X"), "y")
    def test_param(self): self.assertEqual(get_query_param({"a":1},"a"), 1)
    def test_method(self): self.assertTrue(validate_method("get"))
    def test_ua(self): self.assertEqual(get_user_agent({"User-Agent":"x"}), "x")