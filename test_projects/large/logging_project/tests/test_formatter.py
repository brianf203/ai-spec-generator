import unittest
from formatter import format_log, format_with_time, format_simple, format_template, format_compact
class Test(unittest.TestCase):
    def test_format(self): self.assertEqual(format_log("INFO","x"), "[INFO] x")
    def test_time(self): self.assertTrue("[INFO]" in format_with_time("INFO","x","2020-01-01"))
    def test_simple(self): self.assertEqual(format_simple("INFO","x"), "INFO: x")
    def test_template(self): self.assertEqual(format_template("INFO","x"), "INFO - x")
    def test_compact(self): self.assertEqual(format_compact("INFO","x"), "Ix")