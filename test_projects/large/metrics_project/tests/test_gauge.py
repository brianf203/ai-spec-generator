import unittest
from gauge import set_gauge, get_gauge, gauge_keys
class TestGauge(unittest.TestCase):
    def test_set_get(self): g={}; set_gauge(g,"x",5); self.assertEqual(get_gauge(g,"x"), 5)
    def test_keys(self): g={"a":1,"b":2}; self.assertEqual(set(gauge_keys(g)), {"a","b"})