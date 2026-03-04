import unittest
from session import create_session, get_session, destroy_session, session_user, list_sessions
class TestSession(unittest.TestCase):
    def test_create_get(self): s={}; sid=create_session(s,"u1"); self.assertEqual(session_user(s,sid), "u1")
    def test_destroy(self): s={}; sid=create_session(s,"u1"); destroy_session(s,sid); self.assertIsNone(get_session(s,sid))
    def test_list(self): s={}; create_session(s,"u1"); create_session(s,"u2"); self.assertEqual(len(list_sessions(s)), 2)