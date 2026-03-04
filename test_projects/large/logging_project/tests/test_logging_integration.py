import unittest
from formatter import format_log
from handler import filter_logs, log_level
class TestIntegration(unittest.TestCase):
    def test_format_then_filter(self): logs=[format_log("INFO","a"), format_log("ERROR","b")]; self.assertEqual(len(filter_logs(logs,"INFO")), 1)
    def test_level_extract(self): line=format_log("WARNING","msg"); self.assertEqual(log_level(line), "WARNING")