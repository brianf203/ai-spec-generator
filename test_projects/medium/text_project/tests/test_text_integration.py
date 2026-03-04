import unittest
from word_count import word_count, unique_words
from formatting import char_count, remove_punctuation
class TestIntegration(unittest.TestCase):
    def test_word_char(self): s="a b c"; self.assertEqual(word_count(s), 3); self.assertEqual(char_count(s), 5)
    def test_unique_after_punct(self): s="hello, world!"; self.assertGreaterEqual(len(unique_words(remove_punctuation(s))), 1)