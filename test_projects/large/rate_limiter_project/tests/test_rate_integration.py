import unittest
from limiter import check_limit, reset_limit
from window import window_key
class TestIntegration(unittest.TestCase):
    def test_limit_then_reset(self): l={}; check_limit(l,"k",2); check_limit(l,"k",2); ok,_=check_limit(l,"k",2); self.assertFalse(ok); reset_limit(l,"k"); ok,_=check_limit(l,"k",2); self.assertTrue(ok)