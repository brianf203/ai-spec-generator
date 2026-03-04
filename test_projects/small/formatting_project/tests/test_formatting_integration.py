import unittest
from strings import pad_string, truncate_string
from utils import format_percentage, remove_extra_spaces
class TestFmtIntegration(unittest.TestCase):
    def test_pad_truncate(self): self.assertEqual(len(truncate_string(pad_string("x", 5), 3)), 3)
    def test_pct_spaces(self): self.assertEqual(remove_extra_spaces("  a  b  "), "a b")