import unittest
from channel import send_email, get_pending_emails
from template import render_template
class TestIntegration(unittest.TestCase):
    def test_send_rendered(self): c={}; body=render_template("Hi {{name}}", {"name":"A"}); send_email(c,"a@b.com","S",body); self.assertIn("Hi A", get_pending_emails(c)[0]["body"])