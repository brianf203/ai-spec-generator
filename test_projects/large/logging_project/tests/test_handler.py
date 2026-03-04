import unittest
from handler import should_log, filter_logs, filter_by_pattern, log_contains, log_level, count_by_level
class Test(unittest.TestCase):
    def test_should(self): self.assertTrue(should_log("INFO","DEBUG"))
    def test_filter(self): self.assertEqual(filter_logs(["[INFO] a"], "INFO"), ["[INFO] a"])
    def test_pattern(self): self.assertEqual(len(filter_by_pattern(["[INFO] err"], "err")), 1)
    def test_contains(self): self.assertTrue(log_contains("[INFO] hello", "hello"))
    def test_level(self): self.assertEqual(log_level("[INFO] x"), "INFO")
    def test_count(self): self.assertEqual(count_by_level(["[INFO] a","[INFO] b"])["INFO"], 2)