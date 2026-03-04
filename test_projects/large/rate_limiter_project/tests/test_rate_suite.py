import unittest
from limiter import get_current_count, check_limit
from window import is_in_window
class TestSuite(unittest.TestCase):
    def test_count(self): l={}; check_limit(l,"k",5); self.assertEqual(get_current_count(l,"k"), 1)
    def test_window(self): self.assertTrue(is_in_window(50, 0, 60))