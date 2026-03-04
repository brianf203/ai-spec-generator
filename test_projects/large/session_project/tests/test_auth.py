import unittest
from auth import is_authenticated, require_session, session_count
from session import create_session
class TestAuth(unittest.TestCase):
    def test_authenticated(self): s={}; sid=create_session(s,"u"); self.assertTrue(is_authenticated(s,sid))
    def test_count(self): s={}; create_session(s,"u"); self.assertEqual(session_count(s), 1)
    def test_require(self): s={}; sid=create_session(s,"u"); self.assertEqual(require_session(s,sid)["user_id"], "u")