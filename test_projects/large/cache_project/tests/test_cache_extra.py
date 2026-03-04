"""Additional cache tests."""
import unittest
from ttl import ttl_get, ttl_set, ttl_clear
from lru import lru_get, lru_set, lru_clear


class TestCacheExtra(unittest.TestCase):
    def test_ttl_expired(self):
        c = {}
        ttl_set(c, "x", 1, 0)
        self.assertIsNone(ttl_get(c, "x", 1))

    def test_ttl_clear_empty(self):
        c = {}
        ttl_clear(c)
        self.assertEqual(len(c), 0)

    def test_lru_set_get(self):
        c = {}
        lru_set(c, "k", 42, max_size=5)
        self.assertEqual(lru_get(c, "k"), 42)

    def test_lru_clear(self):
        c = {}
        lru_set(c, "x", 1)
        lru_clear(c)
        self.assertIsNone(lru_get(c, "x"))
