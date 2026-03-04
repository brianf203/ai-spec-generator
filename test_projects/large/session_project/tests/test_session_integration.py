import unittest
from session import create_session, destroy_session, list_sessions
from auth import session_count
class TestIntegration(unittest.TestCase):
    def test_create_destroy(self): s={}; sid=create_session(s,"u"); destroy_session(s,sid); self.assertEqual(session_count(s), 0)