import unittest
from validation import validate_username, validate_password
from hash import simple_hash, verify_hash
from session import create_session, session_to_cookie, cookie_to_session
class TestIntegration(unittest.TestCase):
    def test_valid_user_hash(self): u="user1"; self.assertTrue(validate_username(u)); self.assertIsInstance(simple_hash(u), int)
    def test_session_cookie_roundtrip(self): s=create_session(42); c=session_to_cookie(s); s2=cookie_to_session(c); self.assertEqual(s2["user_id"], 42)