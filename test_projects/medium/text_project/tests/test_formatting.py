import unittest
from formatting import char_count, capitalize_sentence, remove_punctuation, normalize_whitespace, indent_lines, wrap_text, truncate_at_word
class Test(unittest.TestCase):
    def test_char(self): self.assertEqual(char_count("hi"), 2)
    def test_cap(self): self.assertEqual(capitalize_sentence("hi"), "Hi")
    def test_remove_punct(self): self.assertEqual(remove_punctuation("a,b!"), "ab")
    def test_normalize(self): self.assertEqual(normalize_whitespace("a  b"), "a b")
    def test_indent(self): self.assertTrue(indent_lines("x", 2).startswith("  "))
    def test_wrap(self): self.assertIn("\n", wrap_text("a b c d", 2))
    def test_truncate(self): self.assertTrue(truncate_at_word("hello world", 8).endswith("..."))