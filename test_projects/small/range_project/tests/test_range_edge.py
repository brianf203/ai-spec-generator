import unittest
from range_ops import range_gap, range_span
from intervals import chunk_range
class TestEdge(unittest.TestCase):
    def test_gap(self): g=range_gap(range(0, 3), range(5, 8)); self.assertEqual(list(g), [3, 4])
    def test_span(self): s=range_span([range(1, 3), range(5, 7)]); self.assertEqual(s.start, 1)
    def test_chunk(self): self.assertEqual(chunk_range(5, 2), [[0, 1], [2, 3], [4]])