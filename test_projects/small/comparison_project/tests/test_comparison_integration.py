import unittest
from compare import min_of_three, clamp_range
from ordering import compare_asc
class TestIntegration(unittest.TestCase):
    def test_min_clamp(self): m=min_of_three(5, 2, 8); self.assertEqual(clamp_range(m, 0, 10), 2)
    def test_compare_min(self): self.assertEqual(compare_asc(min_of_three(1,2,3), 2), -1)