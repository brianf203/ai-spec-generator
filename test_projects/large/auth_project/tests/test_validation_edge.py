import unittest
from validation import validate_password, validate_username, validate_role, check_password_match, sanitize_username
class TestEdge(unittest.TestCase):
    def test_pwd_short(self): self.assertFalse(validate_password("Ab1"))
    def test_username_short(self): self.assertFalse(validate_username("ab"))
    def test_role_invalid(self): self.assertFalse(validate_role("superuser"))
    def test_pwd_match(self): self.assertTrue(check_password_match("a","a"))
    def test_sanitize(self): self.assertTrue(len(sanitize_username("user_123")) <= 32)