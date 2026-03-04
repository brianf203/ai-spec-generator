import unittest
from counter import get_count
from gauge import get_gauge
class TestEdge(unittest.TestCase):
    def test_counter_missing(self): self.assertEqual(get_count({},"x"), 0)
    def test_gauge_missing(self): self.assertIsNone(get_gauge({},"x"))