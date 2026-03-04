import unittest
from session import create_session, get_session
from auth import is_authenticated
class TestSuite(unittest.TestCase):
    def test_create_auth(self): s={}; sid=create_session(s,"u"); self.assertTrue(is_authenticated(s,sid))
    def test_get(self): s={}; sid=create_session(s,"u"); self.assertIsNotNone(get_session(s,sid))