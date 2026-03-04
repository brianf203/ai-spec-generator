import unittest
from session import get_session, session_user
from auth import is_authenticated
class TestEdge(unittest.TestCase):
    def test_missing(self): self.assertIsNone(get_session({},"x"))
    def test_user_missing(self): self.assertIsNone(session_user({},"x"))
    def test_not_authenticated(self): self.assertFalse(is_authenticated({},"x"))