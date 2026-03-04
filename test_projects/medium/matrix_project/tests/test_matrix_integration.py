import unittest
from matrix import matrix_add, matrix_transpose
from utils import create_matrix, matrix_trace
class TestIntegration(unittest.TestCase):
    def test_add_transpose(self): m=[[1,2],[3,4]]; t=matrix_transpose(m); self.assertEqual(matrix_add(m,t)[0][1], 5)
    def test_trace_symmetric(self): m=[[1,2],[2,1]]; self.assertEqual(matrix_trace(matrix_transpose(m)), matrix_trace(m))