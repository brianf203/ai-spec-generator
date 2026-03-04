import unittest
from utils import create_matrix, matrix_copy, matrix_trace
class TestUtils(unittest.TestCase):
    def test_create(self): m=create_matrix(2,3); self.assertEqual(len(m), 2); self.assertEqual(len(m[0]), 3)
    def test_copy(self): m=[[1,2],[3,4]]; c=matrix_copy(m); c[0][0]=99; self.assertEqual(m[0][0], 1)
    def test_trace(self): m=[[1,2],[3,4]]; self.assertEqual(matrix_trace(m), 5)