import unittest
from priority import pq_push, pq_pop, pq_peek, pq_is_empty, pq_size, pq_top_n
class Test(unittest.TestCase):
    def test_push_pop(self): pq=[]; pq_push(pq,"a",2); pq_push(pq,"b",1); self.assertEqual(pq_pop(pq), "b")
    def test_peek(self): pq=[]; pq_push(pq,"a",1); self.assertEqual(pq_peek(pq), "a")
    def test_empty(self): self.assertTrue(pq_is_empty([]))
    def test_size(self): pq=[]; pq_push(pq,"a",1); self.assertEqual(pq_size(pq), 1)
    def test_top_n(self): pq=[]; pq_push(pq,"a",2); pq_push(pq,"b",1); self.assertEqual(pq_top_n(pq,1), ["b"])