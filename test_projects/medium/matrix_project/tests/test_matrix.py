import unittest
from matrix import matrix_rows, matrix_cols, matrix_add, matrix_transpose
class TestMatrix(unittest.TestCase):
    def test_rows_cols(self): m=[[1,2],[3,4]]; self.assertEqual(matrix_rows(m), 2); self.assertEqual(matrix_cols(m), 2)
    def test_add(self): a=[[1,2],[3,4]]; b=[[5,6],[7,8]]; self.assertEqual(matrix_add(a,b), [[6,8],[10,12]])
    def test_transpose(self): m=[[1,2],[3,4]]; self.assertEqual(matrix_transpose(m), [[1,3],[2,4]])