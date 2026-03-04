import unittest
from lru import lru_get, lru_set, lru_delete, lru_has, lru_keys, lru_size, lru_hit_rate
class Test(unittest.TestCase):
    def test_get(self): self.assertIsNone(lru_get({},"x"))
    def test_set(self): c={}; lru_set(c,"x",1); self.assertEqual(c["x"], 1)
    def test_delete(self): c={"x":1}; lru_delete(c,"x"); self.assertNotIn("x", c)
    def test_has(self): self.assertTrue(lru_has({"x":1},"x"))
    def test_keys(self): self.assertEqual(lru_keys({"a":1}), ["a"])
    def test_size(self): self.assertEqual(lru_size({"a":1}), 1)
    def test_hit_rate(self): self.assertEqual(lru_hit_rate(1, 1), 0.5)