import unittest
from fifo import enqueue, dequeue, peek, is_empty, size, enqueue_many, queue_contains
class Test(unittest.TestCase):
    def test_enq_deq(self): q=[]; enqueue(q,1); self.assertEqual(dequeue(q), 1)
    def test_peek(self): q=[1,2]; self.assertEqual(peek(q), 1)
    def test_empty(self): self.assertTrue(is_empty([]))
    def test_size(self): self.assertEqual(size([1,2]), 2)
    def test_many(self): q=[]; enqueue_many(q,[1,2]); self.assertEqual(size(q), 2)
    def test_contains(self): self.assertTrue(queue_contains([1,2], 2))