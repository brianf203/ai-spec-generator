import unittest
from lru import lru_get, lru_set, lru_clear
class TestEdge(unittest.TestCase):
    def test_get_missing(self): self.assertIsNone(lru_get({},"x"))
    def test_set_overwrite(self): c={}; lru_set(c,"x",1); lru_set(c,"x",2); self.assertEqual(c["x"], 2)
    def test_clear(self): c={"x":1}; lru_clear(c); self.assertEqual(len(c), 0)