import unittest
from formatting import reverse_string, capitalize_words, lower_all, upper_all, swap_case
class TestFmtEdge(unittest.TestCase):
    def test_reverse_empty(self): self.assertEqual(reverse_string(""), "")
    def test_reverse_single(self): self.assertEqual(reverse_string("x"), "x")
    def test_swap_mixed(self): self.assertEqual(swap_case("AbC"), "aBc")
    def test_lower_upper(self): self.assertEqual(upper_all(lower_all("Hi")), "HI")