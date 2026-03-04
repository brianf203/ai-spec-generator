import unittest
from limiter import get_current_count, check_limit
class TestEdge(unittest.TestCase):
    def test_missing_key(self): self.assertEqual(get_current_count({},"x"), 0)
    def test_first_request(self): l={}; ok,_=check_limit(l,"x",1); self.assertTrue(ok)