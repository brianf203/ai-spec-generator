import unittest
from heap import heap_push, heap_pop, heapify
class TestIntegration(unittest.TestCase):
    def test_push_then_heapify(self): h=[]; [heap_push(h,x) for x in [3,1,2]]; self.assertEqual(heap_pop(h), 1)
    def test_heapify_sort(self): h=heapify([5,2,8,1]); out=[heap_pop(h) for _ in range(4)]; self.assertEqual(out, [1,2,5,8])