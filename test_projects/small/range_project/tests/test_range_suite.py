import unittest
from range_ops import range_len, range_contains
from intervals import step_range
class TestSuite(unittest.TestCase):
    def test_len(self): self.assertEqual(range_len(range(1, 5)), 4)
    def test_step(self): self.assertEqual(step_range(0, 5, 2), [0, 2, 4])