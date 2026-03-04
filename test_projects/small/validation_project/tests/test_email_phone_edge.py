import unittest
from email_phone import is_valid_email, is_valid_phone, is_valid_url, is_valid_username, is_valid_zip
class TestEmailPhoneEdge(unittest.TestCase):
    def test_email_invalid(self): self.assertFalse(is_valid_email("no-at"))
    def test_phone_invalid(self): self.assertFalse(is_valid_phone("123"))
    def test_url_http(self): self.assertTrue(is_valid_url("http://x.com"))
    def test_zip_nine(self): self.assertTrue(is_valid_zip("123456789"))
    def test_username_short(self): self.assertFalse(is_valid_username("ab"))