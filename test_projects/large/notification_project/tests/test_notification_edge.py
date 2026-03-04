import unittest
from channel import get_pending_emails
from queue import dequeue_notification
class TestEdge(unittest.TestCase):
    def test_empty_emails(self): self.assertEqual(get_pending_emails({}), [])
    def test_dequeue_empty(self): self.assertIsNone(dequeue_notification([]))