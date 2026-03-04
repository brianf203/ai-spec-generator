"""Additional queue tests."""
import unittest
from fifo import enqueue, dequeue, is_empty
from priority import pq_push, pq_pop


class TestQueueExtra(unittest.TestCase):
    def test_fifo_empty(self):
        q = []
        self.assertTrue(is_empty(q))

    def test_fifo_enqueue_dequeue(self):
        q = []
        enqueue(q, 1)
        self.assertEqual(dequeue(q), 1)

    def test_pq_push_pop(self):
        pq = []
        pq_push(pq, "a", 1)
        self.assertEqual(pq_pop(pq), "a")
