import unittest
from session import create_session, is_session_active, invalidate_session, get_session_user, session_metadata
class Test(unittest.TestCase):
    def test_create(self): s=create_session(1); self.assertEqual(s["user_id"], 1)
    def test_active(self): self.assertTrue(is_session_active({"active": True}))
    def test_invalidate(self): s=create_session(1); invalidate_session(s); self.assertFalse(s["active"])
    def test_get_user(self): self.assertEqual(get_session_user({"user_id": 5}), 5)
    def test_metadata(self): self.assertEqual(session_metadata({"user_id":1,"x":2}), {"x":2})