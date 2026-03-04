import unittest
from partition import partition, three_way_partition
class TestEdge(unittest.TestCase):
    def test_partition_single(self): lst=[1]; p=partition(lst,0,0); self.assertEqual(p, 0)
    def test_three_way_single(self): lst=[5]; lt,gt=three_way_partition(lst,0,0); self.assertEqual(lt, 0); self.assertEqual(gt, 0)