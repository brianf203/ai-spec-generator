"""Additional text processing tests."""
import unittest
from word_count import word_count
from formatting import char_count, capitalize_sentence


class TestTextExtra(unittest.TestCase):
    def test_word_count_single(self):
        self.assertEqual(word_count("hello"), 1)

    def test_char_count_empty(self):
        self.assertEqual(char_count(""), 0)

    def test_capitalize_sentence(self):
        self.assertEqual(capitalize_sentence("hello world"), "Hello world")
