import unittest
from email_phone import is_valid_email
from number_validation import is_in_range
from strings import has_min_length
class TestValIntegration(unittest.TestCase):
    def test_email_and_range(self): self.assertTrue(is_valid_email("a@b.com") and is_in_range(5, 0, 10))
    def test_min_len_numeric(self): self.assertTrue(has_min_length("hello", 3))