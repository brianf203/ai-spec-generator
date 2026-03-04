import unittest
from heap import heap_push, heap_pop, heap_peek, heapify
class TestHeap(unittest.TestCase):
    def test_push_pop(self): h=[]; heap_push(h,3); heap_push(h,1); heap_push(h,2); self.assertEqual(heap_pop(h), 1)
    def test_peek(self): h=[1,2,3]; self.assertEqual(heap_peek(h), 1)
    def test_heapify(self): h=heapify([3,1,2]); self.assertEqual(h[0], 1)