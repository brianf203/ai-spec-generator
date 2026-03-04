import unittest
from histogram import observe, get_bucket_count, histogram_buckets
class TestHistogram(unittest.TestCase):
    def test_observe(self): h={}; observe(h,"0-10"); observe(h,"0-10"); self.assertEqual(get_bucket_count(h,"0-10"), 2)
    def test_buckets(self): h={}; observe(h,"c"); observe(h,"a"); self.assertEqual(histogram_buckets(h), ["a","c"])