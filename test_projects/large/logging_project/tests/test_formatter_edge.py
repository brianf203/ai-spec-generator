import unittest
from formatter import format_json, format_with_context, format_multiline
class TestEdge(unittest.TestCase):
    def test_json(self): j=format_json("INFO","x"); self.assertIn("level", j); self.assertIn("msg", j)
    def test_context(self): self.assertIn("ctx", format_with_context("INFO","x", "ctx"))
    def test_multiline(self): self.assertIn("\n", format_multiline("INFO","a\nb"))