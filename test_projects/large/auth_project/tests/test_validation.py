import unittest
from validation import validate_password, validate_username, validate_email, password_strength, validate_role
class Test(unittest.TestCase):
    def test_pwd(self): self.assertTrue(validate_password("Abc12345"))
    def test_user(self): self.assertTrue(validate_username("user1"))
    def test_email(self): self.assertTrue(validate_email("a@b.com"))
    def test_strength(self): self.assertGreaterEqual(password_strength("Abc123!"), 3)
    def test_role(self): self.assertTrue(validate_role("admin"))