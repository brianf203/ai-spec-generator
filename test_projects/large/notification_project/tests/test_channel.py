import unittest
from channel import send_email, send_sms, get_pending_emails, clear_channel
class TestChannel(unittest.TestCase):
    def test_email(self): c={}; send_email(c,"a@b.com","Hi","Hello"); self.assertEqual(len(get_pending_emails(c)), 1)
    def test_sms(self): c={}; send_sms(c,"+1","Hi"); self.assertEqual(len(c.get("sms",[])), 1)
    def test_clear(self): c={"emails":[{}]}; clear_channel(c); self.assertEqual(len(c), 0)