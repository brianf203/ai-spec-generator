import unittest
from fifo import dequeue, peek, is_empty, dequeue_n, queue_from_list
class TestEdge(unittest.TestCase):
    def test_dequeue_empty(self): self.assertIsNone(dequeue([]))
    def test_peek_empty(self): self.assertIsNone(peek([]))
    def test_dequeue_n(self): q=[1,2,3]; self.assertEqual(dequeue_n(q, 2), [1, 2])
    def test_from_list(self): self.assertEqual(queue_from_list([1,2]), [1,2])