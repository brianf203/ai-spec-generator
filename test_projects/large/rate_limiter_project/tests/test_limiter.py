import unittest
from limiter import check_limit, reset_limit, get_current_count
class TestLimiter(unittest.TestCase):
    def test_check_under(self): l={}; ok,c=check_limit(l,"ip1",5); self.assertTrue(ok); self.assertEqual(c, 1)
    def test_check_over(self): l={"ip1":5}; ok,c=check_limit(l,"ip1",5); self.assertFalse(ok)
    def test_reset(self): l={"ip1":3}; reset_limit(l,"ip1"); self.assertEqual(get_current_count(l,"ip1"), 0)