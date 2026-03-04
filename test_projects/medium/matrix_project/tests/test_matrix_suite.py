import unittest
from matrix import matrix_rows, matrix_cols
from utils import matrix_copy
class TestSuite(unittest.TestCase):
    def test_dims(self): m=[[1,2],[3,4]]; self.assertEqual(matrix_rows(m), 2); self.assertEqual(matrix_cols(m), 2)
    def test_copy(self): m=[[1,2]]; c=matrix_copy(m); c[0][0]=99; self.assertEqual(m[0][0], 1)