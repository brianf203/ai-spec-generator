import unittest
from ttl import ttl_get, ttl_set, ttl_delete, ttl_has, ttl_expiry, ttl_remaining
class Test(unittest.TestCase):
    def test_set_get(self): c={}; ttl_set(c,"x",1,999); self.assertEqual(ttl_get(c,"x",0), 1)
    def test_delete(self): c={}; ttl_set(c,"x",1,999); ttl_delete(c,"x"); self.assertIsNone(ttl_get(c,"x",0))
    def test_has(self): c={}; ttl_set(c,"x",1,999); self.assertTrue(ttl_has(c,"x",0))
    def test_expiry(self): c={}; ttl_set(c,"x",1,999); self.assertEqual(ttl_expiry(c,"x"), 999)
    def test_remaining(self): c={}; ttl_set(c,"x",1,100); self.assertEqual(ttl_remaining(c,"x",0), 100)