import unittest
from formatting import reverse_string
from utils import count_chars, first_char, last_char
from validation import is_empty
class TestStrIntegration(unittest.TestCase):
    def test_reverse_count(self): self.assertEqual(count_chars(reverse_string("abc")), 3)
    def test_empty_first(self): self.assertEqual(first_char(""), "")
    def test_empty_check(self): self.assertTrue(is_empty(""))