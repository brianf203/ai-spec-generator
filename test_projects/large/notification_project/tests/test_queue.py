import unittest
from queue import enqueue_notification, dequeue_notification, queue_size
class TestQueue(unittest.TestCase):
    def test_enqueue_dequeue(self): q=[]; enqueue_notification(q,{"t":"e"}); self.assertEqual(dequeue_notification(q)["t"], "e")
    def test_size(self): q=[]; enqueue_notification(q,{}); enqueue_notification(q,{}); self.assertEqual(queue_size(q), 2)