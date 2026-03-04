import unittest
from compare import max_of_three, in_range
class TestSuite(unittest.TestCase):
    def test_max(self): self.assertEqual(max_of_three(1, 3, 2), 3)
    def test_in_range(self): self.assertTrue(in_range(5, 0, 10))