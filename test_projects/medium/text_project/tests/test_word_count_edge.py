import unittest
from word_count import word_count, sentence_count, line_count, avg_word_length, max_word_length
class TestEdge(unittest.TestCase):
    def test_empty(self): self.assertEqual(word_count(""), 0)
    def test_whitespace(self): self.assertEqual(word_count("   "), 0)
    def test_single_word(self): self.assertEqual(word_count("hello"), 1)
    def test_avg_empty(self): self.assertEqual(avg_word_length(""), 0)
    def test_max_empty(self): self.assertEqual(max_word_length(""), 0)