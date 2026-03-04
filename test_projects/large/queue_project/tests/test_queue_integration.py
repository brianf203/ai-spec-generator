import unittest
from fifo import enqueue, dequeue, size
from priority import pq_push, pq_pop
class TestIntegration(unittest.TestCase):
    def test_fifo_order(self): q=[]; enqueue(q,1); enqueue(q,2); self.assertEqual(dequeue(q), 1); self.assertEqual(dequeue(q), 2)
    def test_pq_order(self): pq=[]; pq_push(pq,"low",2); pq_push(pq,"high",1); self.assertEqual(pq_pop(pq), "high")