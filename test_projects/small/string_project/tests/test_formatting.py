import unittest
from formatting import reverse_string, capitalize_words, lower_all, upper_all, swap_case
class TestFmt(unittest.TestCase):
    def test_reverse(self): self.assertEqual(reverse_string("abc"), "cba")
    def test_capitalize(self): self.assertEqual(capitalize_words("hi"), "Hi")
    def test_lower(self): self.assertEqual(lower_all("ABC"), "abc")
    def test_upper(self): self.assertEqual(upper_all("abc"), "ABC")
    def test_swap(self): self.assertEqual(swap_case("Ab"), "aB")