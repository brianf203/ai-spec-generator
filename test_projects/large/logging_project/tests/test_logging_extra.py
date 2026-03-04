"""Additional logging tests."""
import unittest
from formatter import format_log, format_simple
from handler import filter_logs, count_by_level


class TestLoggingExtra(unittest.TestCase):
    def test_format_log(self):
        s = format_log("INFO", "msg")
        self.assertIn("msg", s)

    def test_format_simple(self):
        s = format_simple("INFO", "test")
        self.assertIn("test", s)

    def test_filter_logs(self):
        logs = ["[INFO] a", "[ERROR] b"]
        filtered = filter_logs(logs, "INFO")
        self.assertEqual(len(filtered), 1)
        self.assertIn("INFO", filtered[0])
