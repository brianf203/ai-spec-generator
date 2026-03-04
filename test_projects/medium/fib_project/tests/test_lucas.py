import unittest
from lucas import lucas, lucas_list
class Test(unittest.TestCase):
    def test_lucas(self): self.assertEqual(lucas(5), 11)
    def test_list(self): self.assertEqual(lucas_list(4), [2,1,3,4])