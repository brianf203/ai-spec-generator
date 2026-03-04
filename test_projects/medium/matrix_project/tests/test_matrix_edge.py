import unittest
from matrix import matrix_multiply, matrix_transpose
from utils import create_matrix
class TestEdge(unittest.TestCase):
    def test_multiply(self): a=[[1,2],[3,4]]; b=[[5,6],[7,8]]; r=matrix_multiply(a,b); self.assertEqual(r[0][0], 19)
    def test_empty(self): self.assertEqual(matrix_transpose([]), [])