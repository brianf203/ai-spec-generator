import unittest
from email_phone import is_valid_email, is_valid_phone, is_valid_url, is_valid_username, is_valid_zip
class TestVal(unittest.TestCase):
    def test_email(self): self.assertTrue(is_valid_email("a@b.com"))
    def test_phone(self): self.assertTrue(is_valid_phone("1234567"))
    def test_url(self): self.assertTrue(is_valid_url("https://x.com"))
    def test_username(self): self.assertTrue(is_valid_username("user_1"))
    def test_zip(self): self.assertTrue(is_valid_zip("12345"))