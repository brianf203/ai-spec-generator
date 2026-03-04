import unittest
from lru import lru_set, lru_get, lru_has
from ttl import ttl_set, ttl_get
class TestIntegration(unittest.TestCase):
    def test_lru_roundtrip(self): c={}; lru_set(c,"k",42); self.assertEqual(lru_get(c,"k"), 42)
    def test_ttl_roundtrip(self): c={}; ttl_set(c,"k","v",100); self.assertEqual(ttl_get(c,"k",0), "v")