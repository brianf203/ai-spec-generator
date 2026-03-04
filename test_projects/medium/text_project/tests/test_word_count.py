import unittest
from word_count import word_count, sentence_count, line_count, paragraph_count, avg_word_length, max_word_length, unique_words
class Test(unittest.TestCase):
    def test_wc(self): self.assertEqual(word_count("a b c"), 3)
    def test_sc(self): self.assertEqual(sentence_count("a. b."), 2)
    def test_line(self): self.assertEqual(line_count("a\nb"), 2)
    def test_para(self): self.assertEqual(paragraph_count("a\n\nb"), 2)
    def test_avg(self): self.assertEqual(avg_word_length("ab cd"), 2.0)
    def test_max(self): self.assertEqual(max_word_length("a bc"), 2)
    def test_unique(self): self.assertEqual(len(unique_words("a b a")), 2)