import unittest
from counter import increment, get_count
from gauge import set_gauge, get_gauge
from histogram import observe, get_bucket_count
class TestIntegration(unittest.TestCase):
    def test_counter_gauge(self): c={}; g={}; increment(c,"req"); set_gauge(g,"latency",10); self.assertEqual(get_count(c,"req"), 1); self.assertEqual(get_gauge(g,"latency"), 10)
    def test_histogram(self): h={}; observe(h,"ok"); observe(h,"ok"); self.assertEqual(get_bucket_count(h,"ok"), 2)