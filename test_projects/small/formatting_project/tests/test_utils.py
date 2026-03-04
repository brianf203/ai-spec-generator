import unittest
from utils import format_percentage, remove_extra_spaces, format_currency, format_int, ellipsis
class TestUtils(unittest.TestCase):
    def test_pct(self): self.assertEqual(format_percentage(50), "50.0%")
    def test_spaces(self): self.assertEqual(remove_extra_spaces("a  b"), "a b")
    def test_currency(self): self.assertEqual(format_currency(10.5), "$10.50")
    def test_format_int(self): self.assertEqual(format_int(1000), "1,000")
    def test_ellipsis(self): self.assertEqual(ellipsis("hello", 3), "hel...")